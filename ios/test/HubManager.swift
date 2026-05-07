import CoreBluetooth

private let hubName = "Matt's Hub"
private let serviceUUID = CBUUID(string: "C5F50001-8280-46DA-89F4-6D8051E4AEEF")
private let charUUID    = CBUUID(string: "C5F50002-8280-46DA-89F4-6D8051E4AEEF")
private let cmdStartProgram: UInt8 = 0x01
private let cmdStopProgram:  UInt8 = 0x00
private let cmdWriteStdin:   UInt8 = 0x06
private let flagProgramRunning: UInt8 = 0x40
private let maxStdinPayload = 19

// MARK: - Types

enum HubConnectionState: String {
    case disconnected = "Disconnected"
    case scanning     = "Scanning"
    case connecting   = "Connecting"
    case connected    = "Connected"
    case ready        = "Ready"
}

// MARK: - Pure helpers

func parseStdoutLines(buffer: inout Data, data: Data) -> [String] {
    buffer.append(contentsOf: data)
    var lines: [String] = []
    while let newline = buffer.firstIndex(of: 0x0A) {
        let line = String(data: Data(buffer[buffer.startIndex..<newline]), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        buffer = Data(buffer[(newline + 1)...])
        if !line.isEmpty { lines.append(line) }
    }
    return lines
}

func frameCommand(_ command: String) -> [Data] {
    let payload = (command + "\n").data(using: .utf8)!
    var chunks: [Data] = []
    var offset = 0
    while offset < payload.count {
        let end = min(offset + maxStdinPayload, payload.count)
        var chunk = Data([cmdWriteStdin])
        chunk.append(payload[offset..<end])
        chunks.append(chunk)
        offset = end
    }
    return chunks
}

func parseStatusNotification(_ data: Data) -> Bool? {
    guard data.count >= 2, data[0] != 0x01 else { return nil }
    return (data[1] & flagProgramRunning) != 0
}

// MARK: - HubManager

@Observable
final class HubManager: NSObject, @unchecked Sendable,
                        CBCentralManagerDelegate,
                        CBPeripheralDelegate {

    var onStateChange: ((HubConnectionState) -> Void)?
    var onLine: ((String) -> Void)?

    var state: HubConnectionState = .disconnected {
        didSet { onStateChange?(state) }
    }

    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var characteristic: CBCharacteristic?
    private var stdoutBuffer = Data()
    private var pendingScan = false
    private var pendingWrites = 0
    private var pendingDisconnect = false
    private var shouldAutoReconnect = false
    private var readyTimer: Timer?

    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
    }

    // MARK: - Public API

    func connect() {
        guard state == .disconnected else { return }
        shouldAutoReconnect = true
        state = .scanning
        if central.state == .poweredOn {
            central.scanForPeripherals(withServices: nil, options: nil)
        } else {
            pendingScan = true
        }
    }

    func disconnect() {
        shouldAutoReconnect = false
        central.stopScan()
        pendingScan = false
        guard let peripheral, let characteristic, !pendingDisconnect else {
            if let p = peripheral { central.cancelPeripheralConnection(p) }
            cleanup()
            state = .disconnected
            return
        }
        pendingDisconnect = true
        pendingWrites += 1
        peripheral.writeValue(Data([cmdStopProgram]), for: characteristic, type: .withResponse)
    }

    func sendCommand(_ command: String) {
        guard let characteristic, let peripheral, !pendingDisconnect else { return }
        for chunk in frameCommand(command) {
            pendingWrites += 1
            peripheral.writeValue(chunk, for: characteristic, type: .withResponse)
        }
    }

    // MARK: - CBCentralManagerDelegate

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn && pendingScan {
            pendingScan = false
            central.scanForPeripherals(withServices: nil, options: nil)
        } else if central.state != .poweredOn && state != .disconnected {
            if let p = peripheral { central.cancelPeripheralConnection(p) }
            cleanup()
            state = .disconnected
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let name = peripheral.name ?? advertisementData[CBAdvertisementDataLocalNameKey] as? String
        guard name == hubName else { return }
        central.stopScan()
        self.peripheral = peripheral
        peripheral.delegate = self
        state = .connecting
        central.connect(peripheral, options: nil)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        state = .connected
        peripheral.discoverServices([serviceUUID])
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        cleanup()
        state = .disconnected
        if shouldAutoReconnect { connect() }
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        cleanup()
        state = .disconnected
        if shouldAutoReconnect { connect() }
    }

    // MARK: - CBPeripheralDelegate

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil,
              let service = peripheral.services?.first(where: { $0.uuid == serviceUUID }) else {
            central.cancelPeripheralConnection(peripheral)
            return
        }
        peripheral.discoverCharacteristics([charUUID], for: service)
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard error == nil,
              let char = service.characteristics?.first(where: { $0.uuid == charUUID }) else {
            central.cancelPeripheralConnection(peripheral)
            return
        }
        self.characteristic = char
        peripheral.setNotifyValue(true, for: char)
        pendingWrites += 1
        peripheral.writeValue(Data([cmdStartProgram]), for: char, type: .withResponse)
        startReadyTimer()
    }

    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        pendingWrites = max(0, pendingWrites - 1)
        if pendingDisconnect && pendingWrites == 0 {
            pendingDisconnect = false
            central.cancelPeripheralConnection(peripheral)
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let data = characteristic.value, !data.isEmpty else { return }

        if let programRunning = parseStatusNotification(data) {
            if !programRunning && state == .ready { state = .connected }
            return
        }

        guard data[0] == 0x01 else { return }
        for line in parseStdoutLines(buffer: &stdoutBuffer, data: data.dropFirst()) {
            if line.hasPrefix("telem|") {
                onLine?(line)
                continue
            }
            onLine?(line)
            if line == "event|ready" { 
                readyTimer?.invalidate()
                readyTimer = nil
                state = .ready 
            }
        }
    }

    // MARK: - Private

    private func startReadyTimer() {
        readyTimer?.invalidate()
        readyTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: false) { [weak self] _ in
            guard let self, self.state == .connected else { return }
            self.disconnect()
        }
    }

    private func cleanup() {
        peripheral = nil
        characteristic = nil
        stdoutBuffer = Data()
        pendingWrites = 0
        pendingDisconnect = false
        readyTimer?.invalidate()
        readyTimer = nil
    }
}
