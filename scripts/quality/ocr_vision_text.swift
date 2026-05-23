#!/usr/bin/env swift
import Foundation
import ImageIO
import Vision

struct OCRCandidate: Encodable {
    let text: String
    let confidence: Float
}

struct OCRObservation: Encodable {
    let text: String
    let confidence: Float
    let candidates: [OCRCandidate]
}

struct OCRRow: Encodable {
    let image_path: String
    let text: String
    let observations: [OCRObservation]
}

func loadCGImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        return nil
    }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}

func recognizeText(path: String) throws -> OCRRow {
    guard let image = loadCGImage(path) else {
        return OCRRow(image_path: path, text: "", observations: [])
    }

    var observationsPayload: [OCRObservation] = []
    let request = VNRecognizeTextRequest { request, error in
        if let error = error {
            fputs("OCR error for \(path): \(error.localizedDescription)\n", stderr)
            return
        }
        let observations = (request.results as? [VNRecognizedTextObservation]) ?? []
        for observation in observations {
            let candidates = observation.topCandidates(5).map { candidate in
                OCRCandidate(text: candidate.string, confidence: candidate.confidence)
            }
            if let best = candidates.first {
                observationsPayload.append(
                    OCRObservation(text: best.text, confidence: best.confidence, candidates: candidates)
                )
            }
        }
    }
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.01
    request.recognitionLanguages = ["en-US", "ja-JP"]

    let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
    try handler.perform([request])
    let text = observationsPayload.map { $0.text }.joined(separator: "\n")
    return OCRRow(image_path: path, text: text, observations: observationsPayload)
}

let paths = CommandLine.arguments.dropFirst()
let encoder = JSONEncoder()
encoder.outputFormatting = [.withoutEscapingSlashes]

for path in paths {
    do {
        let row = try recognizeText(path: path)
        let data = try encoder.encode(row)
        if let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    } catch {
        let row = OCRRow(image_path: path, text: "", observations: [])
        let data = try encoder.encode(row)
        if let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    }
}
