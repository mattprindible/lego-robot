//
//  testApp.swift
//  test
//
//  Created by Matt Prindible on 3/24/26.
//

import SwiftUI

@main
struct testApp: App {
    @State private var streamer = CameraStreamer()

    var body: some Scene {
        WindowGroup {
            ContentView(streamer: streamer)
        }
    }
}
