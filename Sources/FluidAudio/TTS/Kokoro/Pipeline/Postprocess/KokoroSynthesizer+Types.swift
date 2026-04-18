import Foundation

extension KokoroSynthesizer {
    /// Voxnotes patch (F3): controls the global peak-normalization stage
    /// at the end of `synthesizeDetailed`.
    ///
    /// - `.fullDivide` (default): divide every sample by `maxMagnitude`
    ///   so the programme peak sits at exactly 1.0. Preserves FluidAudio's
    ///   historical behaviour.
    /// - `.safetyOnly`: divide only when `maxMagnitude > 1.0`, i.e. pure
    ///   true-peak clipping avoidance. Leaves FluidAudio's raw dynamics
    ///   alone so downstream callers (Voxnotes) can own the loudness
    ///   stage end-to-end.
    public enum PeakNormalizationMode: Sendable {
        case fullDivide
        case safetyOnly
    }

    /// Voxnotes patch (F5): preflight output of `estimateChunks`.
    /// Lets a caller check ahead of time how FluidAudio will chunk a
    /// given input text without paying the CoreML inference cost.
    public struct EstimatedChunk: Sendable {
        public let index: Int
        public let text: String
        public let wordCount: Int
        public let tokenCount: Int
        public let pauseAfterMs: Int
        public let variant: ModelNames.TTS.Variant

        public init(
            index: Int,
            text: String,
            wordCount: Int,
            tokenCount: Int,
            pauseAfterMs: Int,
            variant: ModelNames.TTS.Variant
        ) {
            self.index = index
            self.text = text
            self.wordCount = wordCount
            self.tokenCount = tokenCount
            self.pauseAfterMs = pauseAfterMs
            self.variant = variant
        }
    }

    public struct TokenCapacities {
        public let short: Int
        public let long: Int

        public func capacity(for variant: ModelNames.TTS.Variant) -> Int {
            switch variant {
            case .fiveSecond:
                return short
            case .fifteenSecond:
                return long
            }
        }
    }

    public struct SynthesisResult: Sendable {
        public let audio: Data
        public let chunks: [ChunkInfo]
        public let diagnostics: Diagnostics?

        public init(audio: Data, chunks: [ChunkInfo], diagnostics: Diagnostics? = nil) {
            self.audio = audio
            self.chunks = chunks
            self.diagnostics = diagnostics
        }
    }

    public struct Diagnostics: Sendable {
        public let variantFootprints: [ModelNames.TTS.Variant: Int]
        public let lexiconEntryCount: Int
        public let lexiconEstimatedBytes: Int
        public let audioSampleBytes: Int
        public let outputWavBytes: Int

        public func updating(audioSampleBytes: Int, outputWavBytes: Int) -> Diagnostics {
            Diagnostics(
                variantFootprints: variantFootprints,
                lexiconEntryCount: lexiconEntryCount,
                lexiconEstimatedBytes: lexiconEstimatedBytes,
                audioSampleBytes: audioSampleBytes,
                outputWavBytes: outputWavBytes
            )
        }
    }

    public struct ChunkInfo: Sendable {
        public let index: Int
        public let text: String
        public let wordCount: Int
        public let words: [String]
        public let atoms: [String]
        public let pauseAfterMs: Int
        public let tokenCount: Int
        public let samples: [Float]
        public let variant: ModelNames.TTS.Variant

        /// Voxnotes patch (F2): duration-predictor output summed × 600 samples/frame.
        /// This is the sample count Kokoro's duration predictor claimed it would emit.
        /// `samples.count` is `min(predictedSampleCount, raw CoreML audio output length)`.
        /// When `samples.count < predictedSampleCount`, the model self-truncated — a
        /// strong signal of tail-truncation we can diagnose post-hoc.
        /// Zero if pred_dur was unavailable (older model variants).
        public let predictedSampleCount: Int

        public init(
            index: Int,
            text: String,
            wordCount: Int,
            words: [String],
            atoms: [String],
            pauseAfterMs: Int,
            tokenCount: Int,
            samples: [Float],
            variant: ModelNames.TTS.Variant,
            predictedSampleCount: Int = 0
        ) {
            self.index = index
            self.text = text
            self.wordCount = wordCount
            self.words = words
            self.atoms = atoms
            self.pauseAfterMs = pauseAfterMs
            self.tokenCount = tokenCount
            self.samples = samples
            self.variant = variant
            self.predictedSampleCount = predictedSampleCount
        }
    }

    struct ChunkInfoTemplate: Sendable {
        let index: Int
        let text: String
        let wordCount: Int
        let words: [String]
        let atoms: [String]
        let pauseAfterMs: Int
        let tokenCount: Int
        let variant: ModelNames.TTS.Variant
        let targetTokens: Int
    }

    struct ChunkEntry: Sendable {
        let chunk: TextChunk
        let inputIds: [Int32]
        let template: ChunkInfoTemplate
    }
}
