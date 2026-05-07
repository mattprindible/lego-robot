import Foundation
import ARKit
import UIKit
import Observation

// ARHelper owns the ARSession, WebSocket, and HubManager.
// @unchecked Sendable because we control all writes:
// ARKit calls the delegate serially, wsTask is only replaced from connect().
final class ARHelper: NSObject, ARSessionDelegate, @unchecked Sendable {

    var onFrameSent: (() -> Void)?
    let hub = HubManager()

    private let arSession = ARSession()
    private var wsTask: URLSessionWebSocketTask?
    private let urlSession = URLSession(configuration: .default)

    private var lastSentTime: Double = 0
    private let frameInterval: Double = 1.0 / 5.0   // 5 fps
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    func setup() {
        arSession.delegate = self
        let config = ARWorldTrackingConfiguration()
        arSession.run(config)

        // Forward hub output lines to Python over WebSocket
        hub.onLine = { [weak self] line in
            self?.sendJSON(["type": "hub_line", "line": line])
        }

        // Forward hub connection state changes to Python
        hub.onStateChange = { [weak self] state in
            self?.sendJSON(["type": "hub_state", "state": state.rawValue])
        }
    }

    func connect(to urlString: String) {
        wsTask?.cancel(with: .goingAway, reason: nil)
        guard let url = URL(string: urlString) else { return }
        let task = urlSession.webSocketTask(with: url)
        task.resume()
        wsTask = task
        startReceiving(task)
    }

    func ping(completion: @escaping (Bool) -> Void) {
        guard let wsTask else { completion(false); return }
        wsTask.sendPing { error in completion(error == nil) }
    }

    // MARK: - Outgoing helpers

    private func sendJSON(_ msg: [String: Any], completion: (() -> Void)? = nil) {
        guard let data = try? JSONSerialization.data(withJSONObject: msg),
              let text = String(data: data, encoding: .utf8) else {
            completion?()
            return
        }
        if let wsTask = wsTask {
            wsTask.send(.string(text)) { _ in completion?() }
        } else {
            completion?()
        }
    }

    // MARK: - Incoming messages from Python

    private func startReceiving(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self else { return }
            if case .success(let message) = result {
                if case .string(let text) = message { self.handleIncoming(text) }
                self.startReceiving(task)
            } else {
                if self.wsTask === task {
                    self.wsTask = nil
                }
            }
        }
    }

    private func handleIncoming(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        if type == "hub_cmd", let command = json["command"] as? String {
            DispatchQueue.main.async { self.hub.sendCommand(command) }
        } else if type == "hub_connect" {
            DispatchQueue.main.async { self.hub.connect() }
        } else if type == "hub_disconnect" {
            DispatchQueue.main.async { self.hub.disconnect() }
        }
    }

    // MARK: - ARSessionDelegate

    private var isSendingFrame = false

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp
        guard now - lastSentTime >= frameInterval else { return }
        guard !isSendingFrame else { return } // Drop frame if network is backed up
        
        lastSentTime = now
        isSendingFrame = true

        guard let jpeg = Self.encodeJPEG(frame.capturedImage) else { 
            isSendingFrame = false
            return 
        }

        // Pose
        let t = frame.camera.transform
        let pos: [Float] = [t.columns.3.x, t.columns.3.y, t.columns.3.z]
        let q = simd_quaternion(t)
        let rot: [Float] = [q.vector.x, q.vector.y, q.vector.z, q.vector.w]

        let tracking: String
        switch frame.camera.trackingState {
        case .normal:                   tracking = "normal"
        case .limited(let r):
            switch r {
            case .initializing:         tracking = "initializing"
            case .excessiveMotion:      tracking = "excessiveMotion"
            case .insufficientFeatures: tracking = "insufficientFeatures"
            case .relocalizing:         tracking = "relocalizing"
            @unknown default:           tracking = "limited"
            }
        case .notAvailable:             tracking = "unavailable"
        @unknown default:               tracking = "unknown"
        }

        // Camera intrinsics from ARKit (landscape/native sensor coords, before portrait rotation).
        // simd_float3x3 is column-major: columns.0 = first column, etc.
        // Row 0: [fx,  0, cx]  → columns.0.x, columns.1.x, columns.2.x
        // Row 1: [ 0, fy, cy]  → columns.0.y, columns.1.y, columns.2.y
        let intr = frame.camera.intrinsics
        let intrinsics: [Double] = [
            Double(intr.columns.0.x),  // fx
            Double(intr.columns.1.y),  // fy
            Double(intr.columns.2.x),  // cx
            Double(intr.columns.2.y),  // cy
        ]
        let res = frame.camera.imageResolution
        let imageSize: [Double] = [res.width, res.height]  // landscape W×H

        let msg: [String: Any] = [
            "type":       "frame",
            "jpeg":       jpeg.base64EncodedString(),
            "position":   pos.map { Double($0) },
            "rotation":   rot.map { Double($0) },
            "tracking":   tracking,
            "timestamp":  now,
            "intrinsics": intrinsics,   // [fx, fy, cx, cy] in landscape sensor coords
            "image_size": imageSize,    // [width, height] landscape (before portrait rotation)
        ]

        sendJSON(msg) { [weak self] in
            DispatchQueue.main.async {
                self?.isSendingFrame = false
                self?.onFrameSent?()
            }
        }
    }

    private static func encodeJPEG(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.5)
    }
}

// MARK: - CameraStreamer

@Observable
final class CameraStreamer {
    var isConnected = false
    var txFPS: Int = 0
    var trackingState: String = "unavailable"
    var serverURL: String {
        didSet { UserDefaults.standard.set(serverURL, forKey: "serverURL") }
    }

    var hub: HubManager { helper.hub }

    private let helper = ARHelper()
    private var frameCount = 0
    private var fpsWindowStart = Date()

    init() {
        serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? "ws://192.168.0.27:8765"
        start()
    }

    func start() {
        helper.onFrameSent = { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                frameCount += 1
                let elapsed = Date().timeIntervalSince(fpsWindowStart)
                if elapsed >= 1.0 {
                    txFPS = Int(Double(frameCount) / elapsed)
                    frameCount = 0
                    fpsWindowStart = Date()
                }
            }
        }
        helper.setup()
        connect()
        startPingLoop()
    }

    func connect() {
        UserDefaults.standard.set(serverURL, forKey: "serverURL")
        helper.connect(to: serverURL)
    }

    private func startPingLoop() {
        let helper = self.helper
        Task { @MainActor [weak self] in
            while true {
                try? await Task.sleep(for: .seconds(3))
                let ok = await withCheckedContinuation { cont in
                    helper.ping { cont.resume(returning: $0) }
                }
                self?.isConnected = ok
                if !ok, let url = self?.serverURL {
                    helper.connect(to: url)
                }
            }
        }
    }
}
