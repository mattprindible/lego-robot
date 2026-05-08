import Foundation
import AVFoundation
import UIKit
import Observation

// CameraHelper owns the AVCaptureSession, WebSocket, and HubManager.
// @unchecked Sendable: capture callbacks run on captureQueue (serial),
// wsTask is only replaced from connect().
final class CameraHelper: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

    var onFrameSent: (() -> Void)?
    let hub = HubManager()

    private let session = AVCaptureSession()
    private let captureQueue = DispatchQueue(label: "camera.capture", qos: .userInteractive)
    private var wsTask: URLSessionWebSocketTask?
    private let urlSession = URLSession(configuration: .default)

    private var lastSentTime: Double = 0
    private let frameInterval: Double = 1.0 / 5.0  // 5 fps
    private var isSendingFrame = false
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    func setup() {
        setupCamera()
    }

    private func setupCamera() {
        let device = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
                  ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
        guard let input = try? AVCaptureDeviceInput(device: device) else { return }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: captureQueue)

        session.beginConfiguration()
        session.sessionPreset = .medium
        if session.canAddInput(input)  { session.addInput(input) }
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()

        captureQueue.async { self.session.startRunning() }
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

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let now = sampleBuffer.presentationTimeStamp.seconds
        guard now - lastSentTime >= frameInterval, !isSendingFrame else { return }
        lastSentTime = now
        isSendingFrame = true

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let jpeg = Self.encodeJPEG(pixelBuffer) else {
            isSendingFrame = false
            return
        }

        wsTask?.send(.data(jpeg)) { [weak self] _ in
            self?.captureQueue.async {
                self?.isSendingFrame = false
                self?.onFrameSent?()
            }
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
                if self.wsTask === task { self.wsTask = nil }
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
    var serverURL: String {
        didSet { UserDefaults.standard.set(serverURL, forKey: "serverURL") }
    }

    var hub: HubManager { helper.hub }

    private let helper = CameraHelper()
    private var frameCount = 0
    private var fpsWindowStart = Date()

    init() {
        serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? "ws://192.168.0.77:8765"
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
