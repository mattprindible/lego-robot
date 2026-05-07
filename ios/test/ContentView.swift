import SwiftUI

struct ContentView: View {
    @Bindable var streamer: CameraStreamer

    var body: some View {
        VStack(spacing: 24) {
            // Status
            HStack(spacing: 10) {
                Circle()
                    .fill(streamer.isConnected ? Color.green : Color.red)
                    .frame(width: 12, height: 12)
                Text(streamer.isConnected ? "Connected" : "Disconnected")
                    .font(.headline)
                Spacer()
                if streamer.isConnected {
                    Text("\(streamer.txFPS) fps")
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)

            Divider()

            // URL config
            VStack(alignment: .leading, spacing: 8) {
                Text("Laptop WebSocket URL")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("ws://192.168.x.x:8765", text: $streamer.serverURL)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                Button("Connect") { streamer.connect() }
                    .buttonStyle(.borderedProminent)
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal)

            Divider()

            // Hub connection
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Circle()
                        .fill(hubColor(streamer.hub.state))
                        .frame(width: 12, height: 12)
                    Text("Hub: \(streamer.hub.state.rawValue)")
                        .font(.headline)
                }
                HStack {
                    Button("Connect Hub") { streamer.hub.connect() }
                        .buttonStyle(.borderedProminent)
                        .disabled(streamer.hub.state != .disconnected)
                    Button("Disconnect") { streamer.hub.disconnect() }
                        .buttonStyle(.bordered)
                        .disabled(streamer.hub.state == .disconnected)
                }
                .frame(maxWidth: .infinity)
            }
            .padding(.horizontal)

            Spacer()

            Text("Streaming camera + hub to laptop.\nAll logic runs in Python on your Mac.")
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.bottom)
        }
        .padding(.top, 40)
        .onAppear { }
    }

    private func hubColor(_ state: HubConnectionState) -> Color {
        switch state {
        case .disconnected: return .red
        case .scanning, .connecting, .connected: return .orange
        case .ready: return .green
        }
    }
}
