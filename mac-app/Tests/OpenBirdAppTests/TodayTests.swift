import XCTest
@testable import OpenBirdApp

final class TodayTimelineDecodingTests: XCTestCase {
    func testParseDayTimelineDecodesCliJson() {
        let json = """
        {
          "day_offset": 1,
          "start": 1.0,
          "end": 2.0,
          "total_observations": 3,
          "distinct_apps": 2,
          "active_seconds": 120.5,
          "sessions": [
            {"session_id": "s1", "app": "com.google.Chrome", "start": 1.0, "end": 1.5, "count": 2},
            {"session_id": null, "app": "com.apple.finder", "start": 1.6, "end": 1.6, "count": 1}
          ]
        }
        """
        let timeline = OpenBirdService.parseDayTimeline(json)
        XCTAssertEqual(timeline?.dayOffset, 1)
        XCTAssertEqual(timeline?.totalObservations, 3)
        XCTAssertEqual(timeline?.distinctApps, 2)
        XCTAssertEqual(timeline?.activeSeconds, 120.5)
        XCTAssertEqual(timeline?.sessions.count, 2)
        XCTAssertEqual(timeline?.sessions.first?.app, "com.google.Chrome")
        XCTAssertEqual(timeline?.sessions.first?.count, 2)
        XCTAssertNil(timeline?.sessions.last?.sessionId)
    }

    func testParseDayTimelineRejectsNonJson() {
        XCTAssertNil(OpenBirdService.parseDayTimeline("not json"))
    }

    func testParseBriefingExtractsText() {
        let briefing = OpenBirdService.parseBriefing("{\"text\": \"You worked on X.\"}")
        XCTAssertEqual(briefing?.text, "You worked on X.")
        XCTAssertNil(briefing?.reasoningRoute)
        XCTAssertNil(briefing?.routeLabel)
        // No sources key -> empty trail, total defaults to 0 (a stale-CLI briefing).
        XCTAssertEqual(briefing?.sources, [])
        XCTAssertEqual(briefing?.sourcesTotal, 0)
        XCTAssertNil(OpenBirdService.parseBriefing("nope"))
    }

    func testParseBriefingDecodesKnownReasoningRoutes() {
        let local = OpenBirdService.parseBriefing("""
        {"text": "Local facts.", "reasoning_route": "local_deterministic"}
        """)
        XCTAssertEqual(local?.reasoningRoute, "local_deterministic")
        XCTAssertEqual(local?.routeLabel, "Local only")

        let cloud = OpenBirdService.parseBriefing("""
        {"text": "Cloud prose.", "reasoning_route": "cloud_reasoning_active"}
        """)
        XCTAssertEqual(cloud?.routeLabel, "Cloud reasoning active")

        let localModel = OpenBirdService.parseBriefing("""
        {"text": "Local model prose.", "reasoning_route": "local_model"}
        """)
        XCTAssertEqual(localModel?.routeLabel, "Local model")
    }

    func testParseBriefingUnknownReasoningRouteIsNeutral() {
        let briefing = OpenBirdService.parseBriefing("""
        {"text": "Future route.", "reasoning_route": "partial_cloud_review"}
        """)
        XCTAssertEqual(briefing?.reasoningRoute, "partial_cloud_review")
        XCTAssertNil(briefing?.routeLabel)
    }

    func testParseBriefingDecodesSourceTrail() {
        // Mirrors `openbird briefing --json` (cli.py -> select_briefing_sources).
        let json = """
        {
          "day_offset": 0,
          "start": 1.0,
          "end": 2.0,
          "text": "You edited rag.py and took notes.",
          "sources": [
            {"observation_id": "o2", "app": "Code", "window": "rag.py",
             "ts": 1700000060.0, "snippet": "edited rag.py"},
            {"observation_id": "o1", "app": "Notes", "window": null,
             "ts": 1700000000.0, "snippet": "took notes"}
          ],
          "sources_total": 5
        }
        """
        let briefing = OpenBirdService.parseBriefing(json)
        XCTAssertEqual(briefing?.text, "You edited rag.py and took notes.")
        XCTAssertEqual(briefing?.sources.count, 2)
        XCTAssertEqual(briefing?.sourcesTotal, 5)   // capped: 2 of 5 surfaced
        let first = try? XCTUnwrap(briefing?.sources.first)
        XCTAssertEqual(first?.observationId, "o2")
        XCTAssertEqual(first?.window, "rag.py")
        XCTAssertEqual(first?.ts, 1_700_000_060.0)
        XCTAssertNil(briefing?.sources.last?.window)   // null window decodes to nil
    }

    func testParseBriefingEmptyDayHasNoSources() {
        let json = """
        {"day_offset": 0, "start": 1.0, "end": 2.0,
         "text": "[yesterday] No activity recorded in the selected window.",
         "sources": [], "sources_total": 0}
        """
        let briefing = OpenBirdService.parseBriefing(json)
        XCTAssertEqual(briefing?.sources, [])
        XCTAssertEqual(briefing?.sourcesTotal, 0)
    }

    /// A briefing source maps to a `ChatCitation` carrying the navigable fields, so
    /// a clicked source reuses the chat citation navigation unchanged.
    func testBriefingSourceMapsToCitation() {
        let source = BriefingSource(
            observationId: "obs-7", app: "Code", window: "rag.py",
            ts: 1_700_000_000.0, snippet: "edited rag.py"
        )
        let citation = source.asCitation(index: 3)
        XCTAssertEqual(citation.index, 3)
        XCTAssertEqual(citation.observationId, "obs-7")
        XCTAssertNil(citation.chunkId)
        XCTAssertEqual(citation.app, "Code")
        XCTAssertEqual(citation.window, "rag.py")
        XCTAssertEqual(citation.ts, 1_700_000_000.0)
        XCTAssertEqual(citation.snippet, "edited rag.py")
    }
}

final class AppDisplayTests: XCTestCase {
    func testNilOrEmptyBundleIsUnknown() {
        XCTAssertEqual(AppDisplay.name(nil), "Unknown")
        XCTAssertEqual(AppDisplay.name(""), "Unknown")
    }

    func testFallbackCapitalizesLastComponent() {
        // Deterministic (machine-independent) path for an unresolvable bundle id.
        XCTAssertEqual(AppDisplay.fallbackName("com.example.someApp"), "SomeApp")
        XCTAssertEqual(AppDisplay.fallbackName("widget"), "Widget")
    }
}

final class BriefingProseTests: XCTestCase {
    func testTightProseIsSingleParagraph() {
        let raw = "A development-focused day across **GitHub** and the browser."
        XCTAssertEqual(BriefingProse.paragraphs(from: raw), [raw])
    }

    func testStripsHeadingMarkersAndKeepsText() {
        // The verbose CoT dump (### Summary of Observations / ### Final Answer) must
        // never render its `#` markers literally.
        let raw = "### Summary of Observations:\nThe user worked on openbird."
        XCTAssertEqual(
            BriefingProse.paragraphs(from: raw),
            ["Summary of Observations:", "The user worked on openbird."]
        )
    }

    func testStripsHorizontalRulesAndListMarkers() {
        let raw = """
        Overview line.

        ---

        - first item
        1. second item
        """
        XCTAssertEqual(
            BriefingProse.paragraphs(from: raw),
            ["Overview line.", "first item", "second item"]
        )
    }

    func testReflowsWrappedProseLinesIntoOneParagraph() {
        let raw = "A heads-down day\non the citation pipeline."
        XCTAssertEqual(
            BriefingProse.paragraphs(from: raw),
            ["A heads-down day on the citation pipeline."]
        )
    }

    func testStripsBlockquoteMarkers() {
        // `> quoted` must not render a literal blockquote marker (contract: no raw
        // Markdown symbols). Nested `>>` collapses too.
        XCTAssertEqual(
            BriefingProse.paragraphs(from: "> Today was focused on **rag.py**.\n>> nested"),
            ["Today was focused on **rag.py**.", "nested"]
        )
    }

    func testMarkerOnlyLinesProduceNoEmptyParagraphs() {
        // `####`, `### `, and a bare `- ` strip to "" and must be dropped, never
        // appended as blank paragraphs (which would render stray gaps).
        XCTAssertEqual(BriefingProse.paragraphs(from: "####\n### \n- "), [])
        XCTAssertEqual(
            BriefingProse.paragraphs(from: "#### \nReal sentence."),
            ["Real sentence."]
        )
    }

    func testFencedCodeOpenersAreDropped() {
        // ```swift / ~~~json openers must not render literal backticks; fenced content
        // survives as plain prose.
        XCTAssertEqual(
            BriefingProse.paragraphs(from: "```swift\nlet x = 1\n```"),
            ["let x = 1"]
        )
    }

    func testCRLFLineEndingsAreNormalised() {
        // A CRLF tail must not leave `\r` attached and defeat the symbol/heading checks.
        XCTAssertEqual(
            BriefingProse.paragraphs(from: "####\r\nReal sentence.\r\n"),
            ["Real sentence."]
        )
    }

    func testAllSymbolInputStripsToEmpty() {
        // Pure noise normalises to no paragraphs; the BriefingText fallback (not this
        // pure parser) is what keeps the card non-blank.
        XCTAssertEqual(BriefingProse.paragraphs(from: "---\n***\n___"), [])
    }

    func testHashtagIsNotTreatedAsHeading() {
        let raw = "Tagged the note #urgent for review."
        XCTAssertEqual(BriefingProse.paragraphs(from: raw), [raw])
    }

    func testNoBlockMarkersLeakOnRealisticVerboseBriefing() {
        // End-to-end-ish: a realistic multi-section qwen3 dump must reduce to clean
        // paragraphs with NO leaked block markers (the exact failure the bare
        // `Text(briefing)` produced). Inline `**bold**` is preserved for the renderer.
        let raw = """
        The observations reflect activity around the **openbird** repo.

        ### **1. Key Projects and Repositories**
        - **openbird**: A local-first AI memory system. Recent activity includes:
          - **#128**: Added a **Liquid Glass Settings tab**.
          1. Improved the **daily briefing** rendering.

        ---

        ### **2. Notable Features**
        ```swift
        let x = 1
        ```
        """
        let paras = BriefingProse.paragraphs(from: raw)
        XCTAssertFalse(paras.isEmpty)
        for p in paras {
            XCTAssertFalse(p.hasPrefix("#"), "heading marker leaked: \(p)")
            XCTAssertFalse(p.hasPrefix("- "), "list marker leaked: \(p)")
            XCTAssertNotEqual(p, "---")
            XCTAssertFalse(p.hasPrefix("```"), "fence leaked: \(p)")
        }
        // The grounding entities survive as inline-bold-bearing text.
        XCTAssertTrue(paras.contains { $0.contains("**openbird**") })
    }

    func testInlineBoldSurvivesAsAttributedRun() {
        // `**rag.py**` parses to a strongly-emphasised run (no literal asterisks).
        let attr = BriefingProse.inlineAttributed("Worked on **rag.py** today.")
        XCTAssertFalse(String(attr.characters).contains("*"))
        let bolded = attr.runs.contains { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true }
        XCTAssertTrue(bolded)
    }
}

final class TodayFormattingTests: XCTestCase {
    func testDurationLabel() {
        XCTAssertEqual(TodayView.durationLabel(30), "30s")
        XCTAssertEqual(TodayView.durationLabel(90), "1m")
        XCTAssertEqual(TodayView.durationLabel(3661), "1h 1m")
        XCTAssertEqual(TodayView.durationLabel(0), "0s")
    }
}

@MainActor
final class TodayModelTests: XCTestCase {
    func testDayTitleForTodayAndYesterday() {
        let model = TodayModel(service: OpenBirdService())
        model.dayOffset = 0
        XCTAssertEqual(model.dayTitle, "Today")
        model.dayOffset = 1
        XCTAssertEqual(model.dayTitle, "Yesterday")
    }
}
