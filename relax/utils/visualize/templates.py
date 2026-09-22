# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""HTML templates for the Relax rollout result viewer.

Ported from rlsp/utils/visualize. Provides three helpers:
- :func:`get_common_styles` — shared CSS (day/night themes)
- :func:`get_theme_script` — theme switch JS
- :func:`get_jsonl_viewer_html` — single-page rollout_result viewer
"""


def get_common_styles() -> str:
    """Get common CSS styles with day/night themes."""
    return """
    <style>
        :root {
            /* Light theme (default) */
            --bg-primary: #f6f8fa;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-hover: #f3f4f6;
            --text-primary: #1f2937;
            --text-secondary: #4b5563;
            --text-muted: #9ca3af;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --accent-subtle: rgba(37, 99, 235, 0.1);
            --border: #e5e7eb;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --code-bg: #f3f4f6;
            --shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        [data-theme="dark"] {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --bg-hover: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent: #58a6ff;
            --accent-hover: #79c0ff;
            --accent-subtle: rgba(56, 139, 253, 0.15);
            --border: #30363d;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
            --code-bg: #161b22;
            --shadow: 0 8px 24px rgba(0,0,0,0.3);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
            transition: background 0.3s, color 0.3s;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header */
        header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 16px 24px;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            font-size: 1.5rem;
        }

        header h1 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        header .subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 2px;
        }

        /* Theme Toggle */
        .theme-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 4px;
        }

        .theme-btn {
            background: transparent;
            border: none;
            padding: 8px 12px;
            border-radius: 20px;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 1rem;
            transition: all 0.2s;
        }

        .theme-btn.active {
            background: var(--accent-subtle);
            color: var(--accent);
        }

        .theme-btn:hover:not(.active) {
            color: var(--text-primary);
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            transition: box-shadow 0.2s, border-color 0.2s;
        }

        .card:hover {
            border-color: var(--accent);
            box-shadow: var(--shadow);
        }

        .card-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }

        .card-header h2 {
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-body {
            padding: 20px;
        }

        .card.collapsed .card-body {
            display: none;
        }

        .card-header .toggle-icon {
            transition: transform 0.2s;
            color: var(--text-muted);
        }

        .card.collapsed .toggle-icon {
            transform: rotate(-90deg);
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 10px 16px;
            background: var(--accent);
            color: #fff;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.2s;
            border: none;
            cursor: pointer;
        }

        .btn:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }

        .btn-outline:hover {
            background: var(--bg-hover);
            border-color: var(--accent);
            transform: none;
        }

        .btn-sm {
            padding: 6px 12px;
            font-size: 0.8125rem;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Grid */
        .grid {
            display: grid;
            gap: 20px;
        }

        .grid-2 {
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }

        /* Directory Cards */
        .dir-card {
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .dir-icon {
            font-size: 2.5rem;
            margin-bottom: 8px;
        }

        .dir-title {
            font-size: 1.125rem;
            font-weight: 600;
        }

        .dir-desc {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        /* Step Selector - Dropdown Style */
        .step-selector-container {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .step-dropdown {
            padding: 10px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.9375rem;
            cursor: pointer;
            min-width: 150px;
        }

        .step-dropdown:focus {
            outline: none;
            border-color: var(--accent);
        }

        .step-dropdown:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .step-info {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 16px;
            background: var(--accent-subtle);
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .step-info .file-size {
            font-weight: 600;
            color: var(--accent);
        }

        /* Legacy step-grid hidden but kept for compatibility */
        .step-grid {
            display: none;
        }

        /* Controls Row - merged step selector and sort controls */
        .controls-row {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .controls-divider {
            width: 1px;
            height: 24px;
            background: var(--border);
            margin: 0 8px;
        }

        /* Sort Controls */
        .sort-controls {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .sort-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .sort-select {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-size: 0.875rem;
            cursor: pointer;
        }

        .sort-select:focus {
            outline: none;
            border-color: var(--accent);
        }

        /* Sample Navigation - Top bar */
        .sample-nav {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
            padding: 16px 20px;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        .sample-info {
            font-size: 0.9375rem;
            color: var(--text-secondary);
        }

        .sample-info strong {
            color: var(--accent);
            font-weight: 600;
        }

        /* Reward Histogram Styles */
        .histogram-section {
            display: none;
        }

        .histogram-grid {
            display: flex;
            flex-wrap: nowrap;
            gap: 16px;
            overflow-x: auto;
            padding-bottom: 8px;
        }

        .histogram-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px 8px;
            min-width: 200px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .histogram-container {
            height: 100px;
            display: flex;
            align-items: flex-end;
            gap: 3px;
            margin-bottom: 24px;
            position: relative;
        }

        .histogram-bar-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            height: 100%;
            position: relative;
        }

        .histogram-bar-count {
            font-size: 0.625rem;
            color: var(--text-secondary);
            margin-bottom: 2px;
            white-space: nowrap;
            text-align: center;
        }

        .histogram-bar {
            width: 100%;
            background: var(--accent);
            border-radius: 2px 2px 0 0;
            min-height: 2px;
            transition: background 0.2s;
            cursor: pointer;
        }

        .histogram-bar:hover {
            background: var(--accent-hover);
        }

        .histogram-bar-label {
            font-size: 0.6rem;
            color: var(--text-muted);
            text-align: center;
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .histogram-footer {
            text-align: center;
            margin-top: auto;
            padding-top: 4px;
        }

        .histogram-title {
            font-size: 0.8125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .histogram-stats {
            font-size: 0.6875rem;
            color: var(--text-muted);
        }

        /* Floating Navigation Buttons */
        .floating-nav {
            position: fixed;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 12px;
            z-index: 1000;
        }

        .floating-nav .nav-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            font-size: 1.25rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow);
            transition: all 0.2s;
        }

        .floating-nav .nav-btn:hover:not(:disabled) {
            background: var(--accent);
            border-color: var(--accent);
            color: #fff;
            transform: scale(1.1);
        }

        .floating-nav .nav-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .floating-nav .nav-label {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-align: center;
            box-shadow: var(--shadow);
        }

        .floating-nav .nav-label strong {
            color: var(--accent);
            font-weight: 600;
        }

        /* Compact Scalar Table - Multi-column */
        .scalar-table-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 0;
        }

        .scalar-table-cell {
            display: flex;
            align-items: flex-start;
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            border-right: 1px solid var(--border);
            min-width: 0;  /* Allow cell to shrink */
            overflow: hidden;
        }

        .scalar-table-cell:last-child {
            border-right: none;
        }

        .scalar-key {
            color: var(--text-secondary);
            font-weight: 500;
            min-width: 130px;
            flex-shrink: 0;
            font-size: 0.8125rem;
            padding-right: 8px;
        }

        .scalar-value {
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            color: var(--warning);
            font-size: 0.875rem;
            word-break: break-word;
            overflow-wrap: anywhere;
            min-width: 0;
            flex: 1;
        }

        .scalar-value-null {
            color: var(--text-muted);
            font-style: italic;
        }

        .scalar-value-bool-true {
            color: var(--success);
        }

        .scalar-value-bool-false {
            color: var(--error);
        }

        /* Field Display */
        .field-group {
            margin-bottom: 16px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .field-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-secondary);
            cursor: pointer;
            user-select: none;
        }

        .field-header:hover {
            background: var(--bg-hover);
        }

        .field-name {
            font-weight: 600;
            color: var(--accent);
            font-size: 0.9375rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .field-meta {
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--text-muted);
            font-size: 0.8125rem;
        }

        .field-type {
            background: var(--accent-subtle);
            color: var(--accent);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }

        .field-body {
            padding: 16px;
            background: var(--bg-card);
            /* Use clip to hide visually but keep text searchable with Ctrl+F */
            position: absolute;
            clip: rect(0, 0, 0, 0);
            height: 1px;
            width: 1px;
            overflow: hidden;
        }

        .field-group.expanded .field-body {
            position: static;
            clip: auto;
            height: auto;
            width: auto;
            overflow: visible;
        }

        .field-group.expanded .expand-icon {
            transform: rotate(90deg);
        }

        .expand-icon {
            transition: transform 0.2s;
            color: var(--text-muted);
        }

        /* Content Types */
        .content-text-wrapper {
            position: relative;
        }

        .copy-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 4px 10px;
            font-size: 0.75rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-secondary);
            cursor: pointer;
            opacity: 0.7;
            transition: all 0.2s;
            z-index: 5;
        }

        .copy-btn:hover {
            opacity: 1;
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }

        .copy-btn.copied {
            background: var(--success);
            border-color: var(--success);
            color: #fff;
            opacity: 1;
        }

        .content-text {
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.875rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            background: var(--code-bg);
            padding: 16px;
            padding-right: 70px;
            border-radius: 8px;
            max-height: 500px;
            overflow-y: auto;
        }

        /* LLM Tag Syntax Highlighting */
        .llm-tag {
            border-radius: 4px;
            padding: 2px 0;
        }

        /* Think/Reasoning tags - subtle gray background */
        .llm-tag-think {
            color: #6b7280;
            background: rgba(107, 114, 128, 0.1);
        }
        [data-theme="dark"] .llm-tag-think {
            color: #9ca3af;
            background: rgba(156, 163, 175, 0.15);
        }
        .llm-tag-think .llm-tag-name {
            color: #4b5563;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-think .llm-tag-name {
            color: #6b7280;
        }

        /* Tool call tags - orange/amber */
        .llm-tag-tool_call, .llm-tag-tool-call, .llm-tag-function_call {
            color: #d97706;
            background: rgba(217, 119, 6, 0.1);
        }
        [data-theme="dark"] .llm-tag-tool_call,
        [data-theme="dark"] .llm-tag-tool-call,
        [data-theme="dark"] .llm-tag-function_call {
            color: #fbbf24;
            background: rgba(251, 191, 36, 0.15);
        }
        .llm-tag-tool_call .llm-tag-name,
        .llm-tag-tool-call .llm-tag-name,
        .llm-tag-function_call .llm-tag-name {
            color: #b45309;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-tool_call .llm-tag-name,
        [data-theme="dark"] .llm-tag-tool-call .llm-tag-name,
        [data-theme="dark"] .llm-tag-function_call .llm-tag-name {
            color: #f59e0b;
        }

        /* Tools definition - purple */
        .llm-tag-tools {
            color: #7c3aed;
            background: rgba(124, 58, 237, 0.1);
        }
        [data-theme="dark"] .llm-tag-tools {
            color: #a78bfa;
            background: rgba(167, 139, 250, 0.15);
        }
        .llm-tag-tools .llm-tag-name {
            color: #6d28d9;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-tools .llm-tag-name {
            color: #8b5cf6;
        }

        /* Tool result/output - green */
        .llm-tag-tool_result, .llm-tag-tool-result, .llm-tag-output, .llm-tag-result {
            color: #059669;
            background: rgba(5, 150, 105, 0.1);
        }
        [data-theme="dark"] .llm-tag-tool_result,
        [data-theme="dark"] .llm-tag-tool-result,
        [data-theme="dark"] .llm-tag-output,
        [data-theme="dark"] .llm-tag-result {
            color: #34d399;
            background: rgba(52, 211, 153, 0.15);
        }
        .llm-tag-tool_result .llm-tag-name,
        .llm-tag-tool-result .llm-tag-name,
        .llm-tag-output .llm-tag-name,
        .llm-tag-result .llm-tag-name {
            color: #047857;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-tool_result .llm-tag-name,
        [data-theme="dark"] .llm-tag-tool-result .llm-tag-name,
        [data-theme="dark"] .llm-tag-output .llm-tag-name,
        [data-theme="dark"] .llm-tag-result .llm-tag-name {
            color: #10b981;
        }

        /* System/instruction tags - blue */
        .llm-tag-system, .llm-tag-instruction, .llm-tag-context {
            color: #2563eb;
            background: rgba(37, 99, 235, 0.1);
        }
        [data-theme="dark"] .llm-tag-system,
        [data-theme="dark"] .llm-tag-instruction,
        [data-theme="dark"] .llm-tag-context {
            color: #60a5fa;
            background: rgba(96, 165, 250, 0.15);
        }
        .llm-tag-system .llm-tag-name,
        .llm-tag-instruction .llm-tag-name,
        .llm-tag-context .llm-tag-name {
            color: #1d4ed8;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-system .llm-tag-name,
        [data-theme="dark"] .llm-tag-instruction .llm-tag-name,
        [data-theme="dark"] .llm-tag-context .llm-tag-name {
            color: #3b82f6;
        }

        /* User/query tags - cyan */
        .llm-tag-user, .llm-tag-query, .llm-tag-human {
            color: #0891b2;
            background: rgba(8, 145, 178, 0.1);
        }
        [data-theme="dark"] .llm-tag-user,
        [data-theme="dark"] .llm-tag-query,
        [data-theme="dark"] .llm-tag-human {
            color: #22d3ee;
            background: rgba(34, 211, 238, 0.15);
        }
        .llm-tag-user .llm-tag-name,
        .llm-tag-query .llm-tag-name,
        .llm-tag-human .llm-tag-name {
            color: #0e7490;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-user .llm-tag-name,
        [data-theme="dark"] .llm-tag-query .llm-tag-name,
        [data-theme="dark"] .llm-tag-human .llm-tag-name {
            color: #06b6d4;
        }

        /* Assistant/response - teal */
        .llm-tag-assistant, .llm-tag-response, .llm-tag-answer {
            color: #0d9488;
            background: rgba(13, 148, 136, 0.1);
        }
        [data-theme="dark"] .llm-tag-assistant,
        [data-theme="dark"] .llm-tag-response,
        [data-theme="dark"] .llm-tag-answer {
            color: #2dd4bf;
            background: rgba(45, 212, 191, 0.15);
        }
        .llm-tag-assistant .llm-tag-name,
        .llm-tag-response .llm-tag-name,
        .llm-tag-answer .llm-tag-name {
            color: #0f766e;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-assistant .llm-tag-name,
        [data-theme="dark"] .llm-tag-response .llm-tag-name,
        [data-theme="dark"] .llm-tag-answer .llm-tag-name {
            color: #14b8a6;
        }

        /* Error tags - red */
        .llm-tag-error, .llm-tag-exception {
            color: #dc2626;
            background: rgba(220, 38, 38, 0.1);
        }
        [data-theme="dark"] .llm-tag-error,
        [data-theme="dark"] .llm-tag-exception {
            color: #f87171;
            background: rgba(248, 113, 113, 0.15);
        }
        .llm-tag-error .llm-tag-name,
        .llm-tag-exception .llm-tag-name {
            color: #b91c1c;
            font-weight: 600;
        }
        [data-theme="dark"] .llm-tag-error .llm-tag-name,
        [data-theme="dark"] .llm-tag-exception .llm-tag-name {
            color: #ef4444;
        }

        /* Code/JSON blocks inside tags */
        .llm-tag-code, .llm-tag-json {
            color: #be185d;
            background: rgba(190, 24, 93, 0.1);
        }
        [data-theme="dark"] .llm-tag-code,
        [data-theme="dark"] .llm-tag-json {
            color: #f472b6;
            background: rgba(244, 114, 182, 0.15);
        }

        /* Generic unknown tags - neutral */
        .llm-tag-unknown {
            color: #71717a;
            background: rgba(113, 113, 122, 0.1);
        }
        [data-theme="dark"] .llm-tag-unknown {
            color: #a1a1aa;
            background: rgba(161, 161, 170, 0.15);
        }

        /* LLM Tag Content Styling - for content wrapped by tags */
        .llm-tag-content {
            display: inline;
        }

        /* Think content - muted gray */
        .llm-tag-content-think {
            color: #6b7280;
            opacity: 0.85;
        }
        [data-theme="dark"] .llm-tag-content-think {
            color: #9ca3af;
            opacity: 0.8;
        }

        /* Tool call content - slightly muted orange tint */
        .llm-tag-content-tool_call,
        .llm-tag-content-tool-call,
        .llm-tag-content-function_call {
            color: #92400e;
        }
        [data-theme="dark"] .llm-tag-content-tool_call,
        [data-theme="dark"] .llm-tag-content-tool-call,
        [data-theme="dark"] .llm-tag-content-function_call {
            color: #fcd34d;
        }

        /* Tools definition content - purple tint */
        .llm-tag-content-tools {
            color: #5b21b6;
        }
        [data-theme="dark"] .llm-tag-content-tools {
            color: #c4b5fd;
        }

        /* Tool result content - green tint */
        .llm-tag-content-tool_result,
        .llm-tag-content-tool-result,
        .llm-tag-content-result,
        .llm-tag-content-output {
            color: #065f46;
        }
        [data-theme="dark"] .llm-tag-content-tool_result,
        [data-theme="dark"] .llm-tag-content-tool-result,
        [data-theme="dark"] .llm-tag-content-result,
        [data-theme="dark"] .llm-tag-content-output {
            color: #6ee7b7;
        }

        /* Error content - red tint */
        .llm-tag-content-error,
        .llm-tag-content-exception {
            color: #991b1b;
        }
        [data-theme="dark"] .llm-tag-content-error,
        [data-theme="dark"] .llm-tag-content-exception {
            color: #fca5a5;
        }

        /* System/instruction content - blue tint */
        .llm-tag-content-system,
        .llm-tag-content-instruction,
        .llm-tag-content-context {
            color: #1e40af;
        }
        [data-theme="dark"] .llm-tag-content-system,
        [data-theme="dark"] .llm-tag-content-instruction,
        [data-theme="dark"] .llm-tag-content-context {
            color: #93c5fd;
        }

        /* Code/JSON content */
        .llm-tag-content-code,
        .llm-tag-content-json {
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            color: #9d174d;
        }
        [data-theme="dark"] .llm-tag-content-code,
        [data-theme="dark"] .llm-tag-content-json {
            color: #f9a8d4;
        }

        /* Chat Template Special Tokens (e.g., <|im_start|>, <|im_end|>, <|endoftext|>) */
        /* Style consistent with jsonl_templates.py - font color only, no background */
        .chat-token {
            border-radius: 4px;
            padding: 2px 0;
        }

        /* im_start/im_end tokens - cyan */
        .chat-token-im_start, .chat-token-im_end {
            color: #0891b2;
            font-weight: 600;
        }
        [data-theme="dark"] .chat-token-im_start,
        [data-theme="dark"] .chat-token-im_end {
            color: #22d3ee;
        }

        /* vision_start/vision_end tokens - purple */
        .chat-token-vision_start, .chat-token-vision_end {
            color: #7c3aed;
            font-weight: 600;
        }
        [data-theme="dark"] .chat-token-vision_start,
        [data-theme="dark"] .chat-token-vision_end {
            color: #a78bfa;
        }

        /* image_pad token - gray */
        .chat-token-image_pad {
            color: #9ca3af;
            font-weight: 500;
        }
        [data-theme="dark"] .chat-token-image_pad {
            color: #6b7280;
        }

        /* Role tokens after im_start (user, assistant, system) */
        .chat-role {
            font-weight: 600;
            padding: 1px 4px;
            border-radius: 4px;
            margin-left: 2px;
        }

        .chat-role-system {
            color: #2563eb;
            background: rgba(37, 99, 235, 0.1);
        }
        [data-theme="dark"] .chat-role-system {
            color: #60a5fa;
            background: rgba(96, 165, 250, 0.15);
        }

        .chat-role-user {
            color: #0891b2;
            background: rgba(8, 145, 178, 0.1);
        }
        [data-theme="dark"] .chat-role-user {
            color: #22d3ee;
            background: rgba(34, 211, 238, 0.15);
        }

        .chat-role-assistant {
            color: #059669;
            background: rgba(5, 150, 105, 0.1);
        }
        [data-theme="dark"] .chat-role-assistant {
            color: #34d399;
            background: rgba(52, 211, 153, 0.15);
        }

        /* endoftext and other special tokens */
        .chat-token-endoftext, .chat-token-eos, .chat-token-pad {
            color: #71717a;
            background: rgba(113, 113, 122, 0.1);
        }
        [data-theme="dark"] .chat-token-endoftext,
        [data-theme="dark"] .chat-token-eos,
        [data-theme="dark"] .chat-token-pad {
            color: #a1a1aa;
            background: rgba(161, 161, 170, 0.15);
        }

        /* bos/sos tokens - beginning of sequence */
        .chat-token-bos, .chat-token-sos {
            color: #7c3aed;
            background: rgba(124, 58, 237, 0.1);
        }
        [data-theme="dark"] .chat-token-bos,
        [data-theme="dark"] .chat-token-sos {
            color: #a78bfa;
            background: rgba(167, 139, 250, 0.15);
        }

        /* Header tokens - slate/neutral */
        .chat-token-header {
            color: #475569;
            background: rgba(71, 85, 105, 0.1);
        }
        [data-theme="dark"] .chat-token-header {
            color: #94a3b8;
            background: rgba(148, 163, 184, 0.15);
        }

        /* Vision tokens - cyan/teal for visual processing */
        .chat-token-vision {
            color: #0891b2;
            background: rgba(8, 145, 178, 0.1);
        }
        [data-theme="dark"] .chat-token-vision {
            color: #22d3ee;
            background: rgba(34, 211, 238, 0.15);
        }

        /* Image tokens - sky blue */
        .chat-token-image {
            color: #0284c7;
            background: rgba(2, 132, 199, 0.1);
        }
        [data-theme="dark"] .chat-token-image {
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.15);
        }

        /* Video tokens - violet/purple */
        .chat-token-video {
            color: #7c3aed;
            background: rgba(124, 58, 237, 0.1);
        }
        [data-theme="dark"] .chat-token-video {
            color: #a78bfa;
            background: rgba(167, 139, 250, 0.15);
        }

        /* Audio tokens - amber/orange */
        .chat-token-audio {
            color: #d97706;
            background: rgba(217, 119, 6, 0.1);
        }
        [data-theme="dark"] .chat-token-audio {
            color: #fbbf24;
            background: rgba(251, 191, 36, 0.15);
        }

        /* Tool/Function tokens - orange */
        .chat-token-tool {
            color: #ea580c;
            background: rgba(234, 88, 12, 0.1);
        }
        [data-theme="dark"] .chat-token-tool {
            color: #fb923c;
            background: rgba(251, 146, 60, 0.15);
        }

        /* Think/Reasoning tokens - muted gray */
        .chat-token-think {
            color: #6b7280;
            background: rgba(107, 114, 128, 0.1);
        }
        [data-theme="dark"] .chat-token-think {
            color: #9ca3af;
            background: rgba(156, 163, 175, 0.15);
        }

        /* Generic special token fallback */
        .chat-token-generic {
            color: #6366f1;
            background: rgba(99, 102, 241, 0.1);
        }
        [data-theme="dark"] .chat-token-generic {
            color: #818cf8;
            background: rgba(129, 140, 248, 0.15);
        }

        .content-number {
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 1.125rem;
            color: var(--warning);
            font-weight: 600;
        }

        .content-bool {
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: 600;
        }

        .content-bool.true {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
        }

        .content-bool.false {
            background: rgba(239, 68, 68, 0.15);
            color: var(--error);
        }

        /* Array Table */
        .array-table-container {
            overflow-x: auto;
            max-height: 600px;
            overflow-y: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .array-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.8125rem;
        }

        .array-table th {
            background: var(--bg-secondary);
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 1;
        }

        .array-table td {
            padding: 6px 10px;
            border-bottom: 1px solid var(--border);
            color: var(--text-primary);
        }

        .array-table tr:hover td {
            background: var(--bg-hover);
        }

        .array-table .index-col {
            color: var(--text-muted);
            width: 50px;
            text-align: center;
        }

        .array-table .token-col {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: var(--success);
        }

        /* Sortable Table Header */
        .sortable-header {
            cursor: pointer;
            user-select: none;
        }

        .sortable-header:hover {
            background: var(--bg-hover);
        }

        .sort-indicator {
            margin-left: 4px;
            color: var(--accent);
            font-weight: bold;
        }

        /* Decoded Token Cell */
        .decoded-token-cell {
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: var(--success);
            font-size: 0.75rem;
        }

        .decoded-token-cell:hover {
            overflow: visible;
            white-space: normal;
            word-break: break-all;
            background: var(--bg-secondary);
            position: relative;
            z-index: 10;
        }

        /* Agent Multi-Turn Conversation */
        .conversation-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .turn-card {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            background: var(--bg-card);
        }

        .turn-header {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }

        .turn-header:hover {
            background: var(--bg-hover);
        }

        .turn-badge {
            background: var(--accent);
            color: #fff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .turn-title {
            font-weight: 600;
            flex: 1;
        }

        .turn-meta {
            color: var(--text-muted);
            font-size: 0.8125rem;
        }

        .turn-body {
            padding: 16px;
            display: none;
        }

        .turn-card.expanded .turn-body {
            display: block;
        }

        .turn-card.expanded .turn-expand-icon {
            transform: rotate(90deg);
        }

        .turn-expand-icon {
            transition: transform 0.2s;
            color: var(--text-muted);
        }

        .message-block {
            margin-bottom: 16px;
            border-radius: 8px;
            overflow: hidden;
        }

        .message-block:last-child {
            margin-bottom: 0;
        }

        .message-label {
            padding: 8px 12px;
            font-weight: 600;
            font-size: 0.8125rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .message-label.assistant {
            background: var(--accent-subtle);
            color: var(--accent);
        }

        .message-label.tool {
            background: rgba(245, 158, 11, 0.15);
            color: var(--warning);
        }

        .message-label.env {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
        }

        .message-content {
            padding: 12px 16px;
            background: var(--code-bg);
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.8125rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
        }

        .tool-call-block {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid var(--warning);
            border-radius: 8px;
            margin: 8px 0;
            padding: 12px;
        }

        .tool-call-name {
            color: var(--warning);
            font-weight: 600;
            margin-bottom: 8px;
        }

        .tool-call-args {
            background: var(--bg-secondary);
            padding: 8px 12px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.8125rem;
        }

        /* Draggable Fields */
        .field-group.dragging {
            opacity: 0.5;
            border: 2px dashed var(--accent);
        }

        .field-group.drag-over {
            border-top: 3px solid var(--accent);
        }

        .card.dragging {
            opacity: 0.5;
            border: 2px dashed var(--accent);
        }

        .card.drag-over {
            border-top: 3px solid var(--accent);
        }

        .drag-handle {
            cursor: grab;
            color: var(--text-muted);
            padding: 4px;
            margin-right: 8px;
            opacity: 0.6;
            transition: opacity 0.2s;
        }

        .drag-handle:hover {
            opacity: 1;
        }

        .drag-handle:active {
            cursor: grabbing;
        }

        /* Fields Container for drag-and-drop */
        .fields-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* Breadcrumb */
        .breadcrumb {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 24px;
            font-size: 0.875rem;
        }

        .breadcrumb a {
            color: var(--accent);
            text-decoration: none;
        }

        .breadcrumb a:hover {
            text-decoration: underline;
        }

        .breadcrumb .sep {
            color: var(--text-muted);
        }

        .breadcrumb .current {
            color: var(--text-secondary);
        }

        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 40px;
            color: var(--text-secondary);
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Error */
        .error-box {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            padding: 16px 20px;
            border-radius: 8px;
            color: var(--error);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container { padding: 16px; }
            .header-content { flex-direction: column; gap: 12px; }
            .sample-nav { flex-direction: column; gap: 12px; }
        }
    </style>
    """


def get_theme_script() -> str:
    """Get theme switching JavaScript."""
    return """
    <script>
        function initTheme() {
            const saved = localStorage.getItem('relax-theme') || 'dark';
            setTheme(saved);
        }

        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('relax-theme', theme);
            document.querySelectorAll('.theme-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.theme === theme);
            });
        }

        initTheme();
    </script>
    """


def get_jsonl_viewer_html(data_dir: str, base_path: str = "") -> str:
    """Generate the JSONL data viewer page HTML.

    This is a single-page viewer that loads JSONL step files directly,
    without the home page selection flow.

    Args:
        data_dir: Path to the data directory
        base_path: Base URL path for reverse proxy support (e.g., "/absproxy/8080")
    """
    # Normalize base_path for URL construction
    if base_path:
        # Ensure base_path starts with /
        if not base_path.startswith("/"):
            base_path = f"/{base_path}"
        # Ensure base_path ends with /
        if not base_path.endswith("/"):
            base_path = f"{base_path}/"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relax Rollout Result Viewer</title>
        {get_common_styles()}
        <style>
            /* Additional styles for JSONL viewer */
            .step-selector {{
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 20px;
                padding: 16px 20px;
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border);
                flex-wrap: wrap;
            }}

            .subdir-tabs {{
                display: flex;
                gap: 6px;
                margin-bottom: 12px;
            }}

            .subdir-tab {{
                padding: 6px 16px;
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 999px;
                color: var(--text-secondary);
                font-size: 0.875rem;
                cursor: pointer;
                text-transform: capitalize;
            }}

            .subdir-tab:hover {{
                border-color: var(--accent);
                color: var(--text-primary);
            }}

            .subdir-tab.active {{
                background: var(--accent);
                color: white;
                border-color: var(--accent);
            }}

            .step-selector label {{
                font-weight: 600;
                color: var(--text-secondary);
            }}

            .step-dropdown {{
                padding: 8px 16px;
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 8px;
                color: var(--text-primary);
                font-size: 0.9375rem;
                cursor: pointer;
                min-width: 120px;
            }}

            .step-dropdown:focus {{
                outline: none;
                border-color: var(--accent);
            }}

            .stats-bar {{
                display: flex;
                gap: 24px;
                color: var(--text-secondary);
                font-size: 0.875rem;
            }}

            .stats-bar .stat {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}

            .stats-bar .stat-value {{
                color: var(--accent);
                font-weight: 600;
            }}

            /* Score highlight */
            .score-positive {{
                color: var(--success);
                font-weight: 600;
            }}

            .score-negative {{
                color: var(--error);
                font-weight: 600;
            }}

            .score-zero {{
                color: var(--text-muted);
            }}

            /* Completion text with special formatting */
            .completion-text {{
                font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
                font-size: 0.875rem;
                line-height: 1.6;
                white-space: pre-wrap;
                word-break: break-word;
                background: var(--code-bg);
                padding: 16px;
                border-radius: 8px;
                max-height: 500px;
                overflow-y: auto;
            }}

            /* Special LLM token styles - no background, font color only */
            /* im_start/im_end tokens - cyan */
            .llm-token-im {{
                color: #0891b2;
                font-weight: 600;
            }}
            [data-theme="dark"] .llm-token-im {{
                color: #22d3ee;
            }}

            /* vision_start/vision_end tokens - purple */
            .llm-token-vision {{
                color: #7c3aed;
                font-weight: 600;
            }}
            [data-theme="dark"] .llm-token-vision {{
                color: #a78bfa;
            }}

            /* image_pad token - gray */
            .llm-token-pad {{
                color: #9ca3af;
                font-weight: 500;
            }}
            [data-theme="dark"] .llm-token-pad {{
                color: #6b7280;
            }}

            /* Other special tokens - blue */
            .llm-token-other {{
                color: #2563eb;
                font-weight: 600;
            }}
            [data-theme="dark"] .llm-token-other {{
                color: #60a5fa;
            }}

            /* Content inside tags - font color only, no background */
            /* Tools content - purple (same as llm-tag-tools) */
            .llm-tools-content {{
                color: #7c3aed;
                display: inline;
            }}
            [data-theme="dark"] .llm-tools-content {{
                color: #a78bfa;
            }}

            /* Tool call content - orange (same as llm-tag-tool_call) */
            .llm-tool-call-content {{
                color: #d97706;
                display: inline;
            }}
            [data-theme="dark"] .llm-tool-call-content {{
                color: #fbbf24;
            }}

            /* Think content - gray text, no background (matching __init__.py) */
            .llm-think-content {{
                color: #6b7280;
                display: inline;
            }}
            [data-theme="dark"] .llm-think-content {{
                color: #9ca3af;
            }}

            /* Answer content - green */
            .llm-answer-content {{
                color: #059669;
                display: inline;
            }}
            [data-theme="dark"] .llm-answer-content {{
                color: #34d399;
            }}

            /* Metrics table for scalar values */
            .metrics-table {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 12px;
                padding: 16px;
            }}

            .metric-item {{
                display: flex;
                flex-direction: column;
                background: var(--bg-secondary);
                padding: 12px;
                border-radius: 8px;
                border: 1px solid var(--border);
            }}

            .metric-name {{
                font-size: 0.75rem;
                color: var(--text-muted);
                margin-bottom: 4px;
            }}

            .metric-value {{
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text-primary);
            }}

            .metric-value.true {{
                color: var(--success);
            }}

            .metric-value.false {{
                color: var(--error);
            }}
        </style>
    </head>
    <body>
        <header>
            <div class="header-content">
                <div class="logo">
                    <span class="logo-icon">📊</span>
                    <div>
                        <h1>
                            <a href="https://github.com/redai-studio/Relax" target="_blank" rel="noopener"
                               style="color: inherit; text-decoration: none;">Relax</a>
                            Rollout Result Viewer
                        </h1>
                        <div class="subtitle">{data_dir}</div>
                    </div>
                </div>
                <div class="theme-toggle">
                    <a class="theme-btn" href="https://github.com/redai-studio/Relax" target="_blank"
                       rel="noopener" title="Relax on GitHub"
                       style="text-decoration: none; display: inline-flex; align-items: center;">
                        <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor"
                             aria-hidden="true">
                            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                                -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                                .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                                -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27
                                1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07
                                -1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38
                                A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                        </svg>
                    </a>
                    <button class="theme-btn" data-theme="light" onclick="setTheme('light')">☀️</button>
                    <button class="theme-btn" data-theme="dark" onclick="setTheme('dark')">🌙</button>
                </div>
            </div>
        </header>

        <div class="container">
            <!-- Subdir Tabs (train/eval) -->
            <div class="subdir-tabs" id="subdir-tabs" style="display:none;"></div>

            <!-- Step Selector -->
            <div class="step-selector">
                <label for="step-select">📁 Step:</label>
                <select id="step-select" class="step-dropdown" onchange="loadStep(this.value)">
                    <option value="">Loading...</option>
                </select>

                <div class="stats-bar">
                    <div class="stat">
                        <span>Samples:</span>
                        <span class="stat-value" id="sample-count">-</span>
                    </div>
                    <div class="stat">
                        <span>Avg Reward:</span>
                        <span class="stat-value" id="avg-reward">-</span>
                    </div>
                </div>

                <div class="sort-controls" style="margin-left: auto;">
                    <span class="sort-label">Sort by:</span>
                    <select class="sort-select" id="sort-field" onchange="updateSorting()">
                        <option value="sample_index">Index</option>
                        <option value="reward">Reward</option>
                        <option value="prompt_token_count">Prompt Tokens</option>
                        <option value="response_length">Response Length</option>
                        <option value="total_token_count">Total Tokens</option>
                        <option value="image_token_count">Image Tokens</option>
                        <option value="agent_turns">Agent Turns</option>
                    </select>
                    <select class="sort-select" id="sort-order" onchange="updateSorting()">
                        <option value="asc">Ascending</option>
                        <option value="desc">Descending</option>
                    </select>
                </div>
            </div>

            <!-- Sample Navigation -->
            <div class="sample-nav">
                <button class="btn btn-outline btn-sm" id="prev-btn" onclick="prevSample()" disabled>← Previous</button>
                <span class="sample-info" style="margin: 0 20px;">
                    Sample <strong id="current-index">1</strong> of <strong id="total-samples">0</strong>
                </span>
                <button class="btn btn-outline btn-sm" id="next-btn" onclick="nextSample()" disabled>Next →</button>
            </div>

            <!-- Sample Content -->
            <div id="sample-content">
                <div class="loading">
                    <div class="spinner"></div>
                    <span>Loading data...</span>
                </div>
            </div>
        </div>

        <!-- Floating Navigation -->
        <div class="floating-nav" id="floating-nav" style="display: none;">
            <button class="nav-btn" id="floating-prev" onclick="prevSample()">↑</button>
            <div class="nav-label">
                <strong id="floating-current">1</strong> / <span id="floating-total">0</span>
            </div>
            <button class="nav-btn" id="floating-next" onclick="nextSample()">↓</button>
        </div>

        {get_theme_script()}

        <script>
            // ==================== State ====================
            let allSamples = [];
            let sortedIndices = [];
            let currentIndex = 0;
            let currentStep = null;
            let currentSubdir = null;
            let subdirs = [];
            const BASE_PATH = "{base_path}";

            // Hidden fields that should not be displayed
            const HIDDEN_FIELDS = ['_type', 'signature', 'request_id'];

            // Key fields to display prominently (in order)
            const KEY_FIELDS = ['prompt', 'response', 'label'];

            // Scalar fields for compact display (in Sample Info card)
            const SCALAR_FIELDS = [
                'rollout_id', 'sample_index', 'reward', 'prompt_token_count',
                'response_token_count', 'total_token_count', 'image_count',
                'image_token_count', 'multimodal_token_count', 'agent_turns',
                'response_length', 'total_length', 'status', 'group_index', 'dataset'
            ];

            // Metric fields to consolidate into Metrics card (number/boolean flags)
            // This list is for explicit matching; dynamic detection handles other cases
            const METRIC_FIELD_PATTERNS = [
                'loop_count', 'has_loop', 'tool_cnt', 'no_thinking', 'thinking_length',
                'is_valid', 'is_correct', 'has_error', 'reward', 'penalty',
                'turns', 'retry_count', 'total_tokens', 'prompt_tokens', 'completion_tokens',
                'error_calls', 'max_repetitions', 'empty_answer'
            ];

            // ==================== Initialization ====================
            document.addEventListener('DOMContentLoaded', async () => {{
                await loadSubdirs();
            }});

            async function loadSubdirs() {{
                try {{
                    const response = await fetch(buildUrl('/api/jsonl/subdirs'));
                    const data = await response.json();
                    subdirs = data.subdirs || [];
                    if (subdirs.length === 0) {{
                        document.getElementById('sample-content').innerHTML =
                            '<div class="error-box">No subdirectories found in data dir.</div>';
                        return;
                    }}
                    renderSubdirTabs();
                    currentSubdir = subdirs[0];
                    await loadStepList();
                }} catch (error) {{
                    console.error('Failed to load subdirs:', error);
                    document.getElementById('sample-content').innerHTML =
                        '<div class="error-box">Failed to load subdirs: ' + error.message + '</div>';
                }}
            }}

            function renderSubdirTabs() {{
                const container = document.getElementById('subdir-tabs');
                if (subdirs.length <= 1) {{ container.style.display = 'none'; return; }}
                container.style.display = 'flex';
                container.innerHTML = subdirs.map(s =>
                    `<button class="subdir-tab" data-subdir="${{s}}" onclick="switchSubdir('${{s}}')">${{s}}</button>`
                ).join('');
                updateSubdirTabActive();
            }}

            function updateSubdirTabActive() {{
                document.querySelectorAll('.subdir-tab').forEach(btn => {{
                    btn.classList.toggle('active', btn.dataset.subdir === currentSubdir);
                }});
            }}

            async function switchSubdir(subdir) {{
                if (subdir === currentSubdir) return;
                currentSubdir = subdir;
                updateSubdirTabActive();
                await loadStepList();
            }}

            // Helper function to build URLs with base path
            function buildUrl(path) {{
                // Normalize the path
                let normalizedPath = path;
                // Ensure path starts with /
                if (!normalizedPath.startsWith('/')) {{
                    normalizedPath = '/' + normalizedPath;
                }}
                // If BASE_PATH is empty, just return the path
                if (!BASE_PATH) {{
                    return normalizedPath;
                }}
                // Ensure BASE_PATH ends with / if it's not empty
                const base = BASE_PATH.endsWith('/') ? BASE_PATH : BASE_PATH + '/';
                // Remove leading slash from path to avoid double slash
                const cleanPath = normalizedPath.startsWith('/') ? normalizedPath.slice(1) : normalizedPath;
                return base + cleanPath;
            }}

            async function loadStepList() {{
                try {{
                    const response = await fetch(buildUrl(`/api/jsonl/${{currentSubdir}}/steps`));
                    const data = await response.json();

                    const select = document.getElementById('step-select');
                    select.innerHTML = '';

                    if (data.steps.length === 0) {{
                        select.innerHTML = '<option value="">No steps found</option>';
                        return;
                    }}

                    data.steps.forEach(step => {{
                        const option = document.createElement('option');
                        option.value = step.filename;
                        option.textContent = `Step ${{step.step}} (${{formatSize(step.size_bytes)}})`;
                        select.appendChild(option);
                    }});

                    // Load first step
                    loadStep(data.steps[0].filename);
                }} catch (error) {{
                    console.error('Failed to load step list:', error);
                    document.getElementById('sample-content').innerHTML =
                        '<div class="error-box">Failed to load step list: ' + error.message + '</div>';
                }}
            }}

            async function loadStep(filename) {{
                if (!filename) return;

                currentStep = filename;
                document.getElementById('sample-content').innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <span>Loading ${{filename}}...</span>
                    </div>
                `;

                try {{
                    const response = await fetch(buildUrl(`/api/jsonl/${{currentSubdir}}/file/${{filename}}`));
                    const data = await response.json();

                    allSamples = data.samples || [];
                    sortedIndices = allSamples.map((_, i) => i);

                    // Update stats
                    document.getElementById('sample-count').textContent = allSamples.length;

                    if (allSamples.length > 0) {{
                        const avgReward = allSamples.reduce((sum, s) => sum + (s.reward || 0), 0) / allSamples.length;
                        document.getElementById('avg-reward').textContent = avgReward.toFixed(4);
                    }}

                    // Apply current sorting
                    updateSorting();

                    currentIndex = 0;
                    renderCurrentSample();
                    updateNavigation();
                }} catch (error) {{
                    console.error('Failed to load step:', error);
                    document.getElementById('sample-content').innerHTML =
                        '<div class="error-box">Failed to load step: ' + error.message + '</div>';
                }}
            }}

            function updateSorting() {{
                const field = document.getElementById('sort-field').value;
                const order = document.getElementById('sort-order').value;

                sortedIndices = allSamples.map((_, i) => i);

                sortedIndices.sort((a, b) => {{
                    const va = allSamples[a][field] || 0;
                    const vb = allSamples[b][field] || 0;
                    return order === 'asc' ? va - vb : vb - va;
                }});

                currentIndex = 0;
                renderCurrentSample();
                updateNavigation();
            }}

            // ==================== Rendering ====================
            function renderCurrentSample() {{
                if (allSamples.length === 0) {{
                    document.getElementById('sample-content').innerHTML =
                        '<div class="error-box">No samples found in this step.</div>';
                    return;
                }}

                const sampleIndex = sortedIndices[currentIndex];
                const sample = allSamples[sampleIndex];

                let html = '<div class="fields-container" id="fields-container">';

                // Layout order: Step/SampleInfo -> Prompt -> Completion -> Solution -> Metrics -> Extra Info -> Other

                // 1. Sample Info (scalar fields)
                const scalarFields = [];
                for (const field of SCALAR_FIELDS) {{
                    if (field in sample && !HIDDEN_FIELDS.includes(field)) {{
                        scalarFields.push([field, sample[field]]);
                    }}
                }}

                if (scalarFields.length > 0) {{
                    html += renderScalarTable(scalarFields);
                }}

                // 2. Prompt
                if ('prompt' in sample && sample['prompt']) {{
                    html += renderTextField('prompt', sample['prompt']);
                }}

                // 3. Response
                if ('response' in sample && sample['response']) {{
                    html += renderTextField('response', sample['response']);
                }}

                // 4. Label (ground truth)
                if ('label' in sample && sample['label']) {{
                    html += renderTextField('label', sample['label']);
                }}

                // 5. Metrics card (consolidated number/boolean flags)
                const metricFields = collectMetricFields(sample);
                if (metricFields.length > 0) {{
                    html += renderMetricsCard(metricFields);
                }}

                // 6. Extra info (if present)
                if (sample.extra_info && typeof sample.extra_info === 'object') {{
                    html += renderExtraInfo(sample.extra_info);
                }}

                // 7. Metadata (if present and not null)
                if (sample.metadata && sample.metadata !== 'None' && sample.metadata !== null) {{
                    html += renderField('metadata', sample.metadata);
                }}

                // 8. Other fields
                for (const [key, value] of Object.entries(sample)) {{
                    if (HIDDEN_FIELDS.includes(key)) continue;
                    if (SCALAR_FIELDS.includes(key)) continue;
                    if (KEY_FIELDS.includes(key)) continue;
                    if (key === 'extra_info' || key === 'metadata') continue;
                    if (isMetricField(key, value)) continue;

                    html += renderField(key, value);
                }}

                html += '</div>';

                document.getElementById('sample-content').innerHTML = html;

                // Scroll long text fields (prompt, completion) to the end
                scrollTextFieldsToEnd();
            }}

            function scrollTextFieldsToEnd() {{
                // Find all content-text divs in prompt and completion fields
                const textContainers = document.querySelectorAll('[data-field-id="field-prompt"] .content-text, [data-field-id="field-response"] .content-text');
                textContainers.forEach(container => {{
                    // Scroll to the bottom
                    container.scrollTop = container.scrollHeight;
                }});
            }}

            function isMetricField(key, value) {{
                // Check if field is a metric field (number or boolean flag)
                if (METRIC_FIELD_PATTERNS.includes(key)) return true;

                // Flexible detection: any single boolean value is a metric
                if (typeof value === 'boolean') return true;

                // Flexible detection: any single number value (not in other cards) is a metric
                // This handles dynamic fields like error_calls, max_repetitions, etc.
                if (typeof value === 'number' && Number.isFinite(value)) return true;

                return false;
            }}

            function collectMetricFields(sample) {{
                const metrics = [];
                for (const [key, value] of Object.entries(sample)) {{
                    if (HIDDEN_FIELDS.includes(key)) continue;
                    if (SCALAR_FIELDS.includes(key)) continue;
                    if (KEY_FIELDS.includes(key)) continue;
                    if (key === 'extra_info' || key === 'metadata') continue;

                    if (isMetricField(key, value)) {{
                        metrics.push([key, value]);
                    }}
                }}
                return metrics;
            }}

            function renderMetricsCard(metricFields) {{
                let items = metricFields.map(([key, value]) => {{
                    let displayValue = '';
                    let valueClass = 'metric-value';

                    if (value === null || value === undefined) {{
                        displayValue = 'null';
                    }} else if (typeof value === 'boolean') {{
                        displayValue = value ? '✓' : '✗';
                        valueClass += value ? ' true' : ' false';
                    }} else if (typeof value === 'number') {{
                        displayValue = formatNumber(value);
                    }} else {{
                        displayValue = String(value);
                    }}

                    return `<div class="metric-item">
                        <span class="metric-name">${{key}}</span>
                        <span class="${{valueClass}}">${{displayValue}}</span>
                    </div>`;
                }}).join('');

                return `
                    <div class="card" style="margin-bottom: 16px;" data-field-id="metrics-info">
                        <div class="card-header" onclick="toggleCard(this.parentElement, 'metrics-info')">
                            <h2>
                                <span class="drag-handle" onclick="event.stopPropagation()">⋮⋮</span>
                                📊 Metrics
                            </h2>
                            <span class="toggle-icon">▼</span>
                        </div>
                        <div class="card-body" style="padding: 0;">
                            <div class="metrics-table">${{items}}</div>
                        </div>
                    </div>
                `;
            }}

            function renderScalarTable(fields) {{
                let cells = fields.map(([key, value]) => {{
                    let displayValue = '';
                    if (value === null || value === undefined || value === 'None') {{
                        displayValue = '<span style="color: var(--text-muted)">null</span>';
                    }} else if (typeof value === 'boolean') {{
                        displayValue = `<span class="content-bool ${{value}}">${{value}}</span>`;
                    }} else if (typeof value === 'number') {{
                        // Special formatting for score
                        if (key === 'score') {{
                            const cls = value > 0 ? 'score-positive' : (value < 0 ? 'score-negative' : 'score-zero');
                            displayValue = `<span class="${{cls}}">${{formatNumber(value)}}</span>`;
                        }} else if (key === 'acc') {{
                            displayValue = `<span style="color: var(--success);">${{(value * 100).toFixed(1)}}%</span>`;
                        }} else {{
                            displayValue = formatNumber(value);
                        }}
                    }} else {{
                        displayValue = escapeHtml(String(value));
                    }}
                    return `<div class="scalar-table-cell">
                        <span class="scalar-key">${{key}}</span>
                        <span class="scalar-value">${{displayValue}}</span>
                    </div>`;
                }}).join('');

                return `
                    <div class="card" style="margin-bottom: 16px;" data-field-id="sample-info" draggable="true">
                        <div class="card-header" onclick="toggleCard(this.parentElement, 'sample-info')">
                            <h2>
                                <span class="drag-handle" onclick="event.stopPropagation()">⋮⋮</span>
                                📋 Sample Info
                            </h2>
                            <span class="toggle-icon">▼</span>
                        </div>
                        <div class="card-body" style="padding: 0;">
                            <div class="scalar-table-grid">${{cells}}</div>
                        </div>
                    </div>
                `;
            }}

            function renderExtraInfo(extraInfo) {{
                let cells = Object.entries(extraInfo).map(([key, value]) => {{
                    let displayValue = '';
                    if (value === null || value === undefined) {{
                        displayValue = '<span style="color: var(--text-muted)">null</span>';
                    }} else if (typeof value === 'string' && value.length > 100) {{
                        displayValue = `<span title="${{escapeHtml(value)}}">${{escapeHtml(value.substring(0, 100))}}...</span>`;
                    }} else {{
                        displayValue = escapeHtml(String(value));
                    }}
                    return `<div class="scalar-table-cell">
                        <span class="scalar-key">${{key}}</span>
                        <span class="scalar-value">${{displayValue}}</span>
                    </div>`;
                }}).join('');

                return `
                    <div class="card" style="margin-bottom: 16px;" data-field-id="extra-info">
                        <div class="card-header" onclick="toggleCard(this.parentElement, 'extra-info')">
                            <h2>
                                <span class="drag-handle" onclick="event.stopPropagation()">⋮⋮</span>
                                📎 Extra Info
                            </h2>
                            <span class="toggle-icon">▼</span>
                        </div>
                        <div class="card-body" style="padding: 0;">
                            <div class="scalar-table-grid">${{cells}}</div>
                        </div>
                    </div>
                `;
            }}

            function renderTextField(name, text) {{
                const displayText = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
                const icon = name === 'prompt' ? '📝' : (name === 'response' ? '🤖' : '✅');
                const title = name.charAt(0).toUpperCase() + name.slice(1);

                return `
                    <div class="field-group expanded" data-field-id="field-${{name}}">
                        <div class="field-header" onclick="toggleField(this.parentElement, 'field-${{name}}')">
                            <div class="field-name">
                                <span class="expand-icon">▶</span>
                                ${{icon}} ${{title}}
                            </div>
                            <div class="field-meta">
                                <span class="field-type">string</span>
                                <span>${{displayText.length}} chars</span>
                            </div>
                        </div>
                        <div class="field-body">
                            <div class="content-text">${{highlightLLMTags(escapeHtml(displayText), name)}}</div>
                        </div>
                    </div>
                `;
            }}

            function renderField(name, value) {{
                const {{ type, content, meta }} = formatValue(value);

                return `
                    <div class="field-group" data-field-id="field-${{name}}">
                        <div class="field-header" onclick="toggleField(this.parentElement, 'field-${{name}}')">
                            <div class="field-name">
                                <span class="expand-icon">▶</span>
                                ${{name}}
                            </div>
                            <div class="field-meta">
                                <span class="field-type">${{type}}</span>
                                ${{meta}}
                            </div>
                        </div>
                        <div class="field-body">
                            ${{content}}
                        </div>
                    </div>
                `;
            }}

            function formatValue(value) {{
                if (value === null || value === undefined || value === 'None') {{
                    return {{ type: 'null', content: '<span style="color: var(--text-muted)">null</span>', meta: '' }};
                }}

                if (typeof value === 'boolean') {{
                    return {{
                        type: 'bool',
                        content: `<span class="content-bool ${{value}}">${{value}}</span>`,
                        meta: ''
                    }};
                }}

                if (typeof value === 'number') {{
                    return {{
                        type: 'number',
                        content: `<span class="content-number">${{formatNumber(value)}}</span>`,
                        meta: ''
                    }};
                }}

                if (typeof value === 'string') {{
                    return {{
                        type: 'string',
                        content: `<div class="content-text">${{highlightLLMTags(escapeHtml(value))}}</div>`,
                        meta: `${{value.length}} chars`
                    }};
                }}

                if (Array.isArray(value)) {{
                    const preview = value.slice(0, 10).map(v =>
                        typeof v === 'object' ? JSON.stringify(v) : formatNumber(v)
                    ).join(', ');
                    return {{
                        type: 'array',
                        content: `<div class="content-text">[${{preview}}${{value.length > 10 ? ', ...' : ''}}]</div>`,
                        meta: `${{value.length}} items`
                    }};
                }}

                if (typeof value === 'object') {{
                    const entries = Object.entries(value);
                    const content = entries.map(([k, v]) => {{
                        const {{ content: c }} = formatValue(v);
                        return `<div style="margin-bottom: 8px;"><strong style="color: var(--accent);">${{k}}:</strong> ${{c}}</div>`;
                    }}).join('');
                    return {{
                        type: 'object',
                        content: content || '<span style="color: var(--text-muted)">Empty object</span>',
                        meta: `${{entries.length}} keys`
                    }};
                }}

                return {{ type: 'unknown', content: escapeHtml(String(value)), meta: '' }};
            }}

            // ==================== Utilities ====================
            function formatNumber(n) {{
                if (typeof n !== 'number') return String(n);
                if (Number.isInteger(n)) return n.toString();
                if (Math.abs(n) < 0.0001 || Math.abs(n) > 10000) return n.toExponential(4);
                return n.toFixed(4);
            }}

            function formatSize(bytes) {{
                if (bytes < 1024) return bytes + ' B';
                if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
                return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
            }}

            function escapeHtml(text) {{
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}

            function highlightLLMTags(text, fieldName = '') {{
                // 1. Highlight special LLM tokens with different colors by category

                // im_start/im_end tokens - cyan
                const imTokenPattern = /&lt;\\|(im_start|im_end)[^&]*\\|&gt;/gi;
                text = text.replace(imTokenPattern, (match) => {{
                    return `<span class="llm-token-im">${{match}}</span>`;
                }});

                // vision_start/vision_end tokens - purple
                const visionTokenPattern = /&lt;\\|(vision_start|vision_end)[^&]*\\|&gt;/gi;
                text = text.replace(visionTokenPattern, (match) => {{
                    return `<span class="llm-token-vision">${{match}}</span>`;
                }});

                // image_pad token - gray
                const imagePadPattern = /&lt;\\|image_pad\\|&gt;/gi;
                text = text.replace(imagePadPattern, (match) => {{
                    return `<span class="llm-token-pad">${{match}}</span>`;
                }});

                // Other special tokens (endoftext, pad, sep, unk, mask, bos, eos) - blue
                const otherTokenPattern = /&lt;\\|(endoftext|pad|sep|unk|mask|bos|eos)[^&]*\\|&gt;/gi;
                text = text.replace(otherTokenPattern, (match) => {{
                    return `<span class="llm-token-other">${{match}}</span>`;
                }});

                // 2. Highlight content inside <tools>...</tools> tags (same color as tag)
                const toolsContentPattern = /(&lt;tools&gt;)([\\s\\S]*?)(&lt;\\/tools&gt;)/gi;
                text = text.replace(toolsContentPattern, (match, open, content, close) => {{
                    return `<span class="llm-tag llm-tag-tools">${{open}}</span><span class="llm-tools-content">${{content}}</span><span class="llm-tag llm-tag-tools">${{close}}</span>`;
                }});

                // 3. Highlight content inside <tool_call>...</tool_call> tags (same color as tag)
                const toolCallContentPattern = /(&lt;tool_call&gt;)([\\s\\S]*?)(&lt;\\/tool_call&gt;)/gi;
                text = text.replace(toolCallContentPattern, (match, open, content, close) => {{
                    return `<span class="llm-tag llm-tag-tool_call">${{open}}</span><span class="llm-tool-call-content">${{content}}</span><span class="llm-tag llm-tag-tool_call">${{close}}</span>`;
                }});

                // 4. Highlight content inside <answer>...</answer> tags (green color)
                const answerContentPattern = /(&lt;answer&gt;)([\\s\\S]*?)(&lt;\\/answer&gt;)/gi;
                text = text.replace(answerContentPattern, (match, open, content, close) => {{
                    return `<span class="llm-answer-content">${{open}}</span><span class="llm-answer-content">${{content}}</span><span class="llm-answer-content">${{close}}</span>`;
                }});

                // 5. Gray highlight for <think>...</think> content
                // Handle both complete and incomplete think tags (for Completion field without opening <think>)
                if (fieldName === 'response') {{
                    // First try complete tags
                    const thinkCompletePattern = /(&lt;think&gt;)([\\s\\S]*?)(&lt;\\/think&gt;)/gi;
                    if (thinkCompletePattern.test(text)) {{
                        text = text.replace(/(&lt;think&gt;)([\\s\\S]*?)(&lt;\\/think&gt;)/gi, (match, open, content, close) => {{
                            return `<span class="llm-tag llm-tag-think">${{open}}</span><span class="llm-think-content">${{content}}</span><span class="llm-tag llm-tag-think">${{close}}</span>`;
                        }});
                    }} else {{
                        // Handle case where Completion starts mid-thinking (only has </think>)
                        const thinkClosePattern = /^([\\s\\S]*?)(&lt;\\/think&gt;)/i;
                        if (thinkClosePattern.test(text)) {{
                            text = text.replace(thinkClosePattern, (match, content, close) => {{
                                return `<span class="llm-think-content">${{content}}</span><span class="llm-tag llm-tag-think">${{close}}</span>`;
                            }});
                        }}
                    }}
                }} else {{
                    // For other fields, just highlight complete think tags
                    const thinkPattern = /(&lt;think&gt;)([\\s\\S]*?)(&lt;\\/think&gt;)/gi;
                    text = text.replace(thinkPattern, (match, open, content, close) => {{
                        return `<span class="llm-tag llm-tag-think">${{open}}</span><span class="llm-think-content">${{content}}</span><span class="llm-tag llm-tag-think">${{close}}</span>`;
                    }});
                }}

                // 6. Highlight remaining XML-style LLM tags (that weren't already processed)
                const tagPattern = /&lt;(\\/?)(tool_result|system|user|assistant|error|output|result|function_call)(&gt;|\\s[^&]*&gt;)/gi;
                text = text.replace(tagPattern, (match, slash, tagName, rest) => {{
                    const normalizedTag = tagName.toLowerCase().replace(/-/g, '_');
                    return `<span class="llm-tag llm-tag-${{normalizedTag}}"><span class="llm-tag-name">&lt;${{slash}}${{tagName}}</span>${{rest.replace('&gt;', '')}}&gt;</span>`;
                }});

                return text;
            }}

            // ==================== UI Interactions ====================
            function toggleCard(card, fieldId) {{
                card.classList.toggle('collapsed');
            }}

            function toggleField(group, fieldId) {{
                group.classList.toggle('expanded');
            }}

            function updateNavigation() {{
                const total = allSamples.length;
                document.getElementById('current-index').textContent = currentIndex + 1;
                document.getElementById('total-samples').textContent = total;
                document.getElementById('floating-current').textContent = currentIndex + 1;
                document.getElementById('floating-total').textContent = total;
                document.getElementById('prev-btn').disabled = currentIndex === 0;
                document.getElementById('next-btn').disabled = currentIndex >= total - 1;
                document.getElementById('floating-prev').disabled = currentIndex === 0;
                document.getElementById('floating-next').disabled = currentIndex >= total - 1;

                // Show floating nav when samples are loaded
                document.getElementById('floating-nav').style.display = total > 0 ? 'flex' : 'none';
            }}

            function prevSample() {{
                if (currentIndex > 0) {{
                    currentIndex--;
                    renderCurrentSample();
                    updateNavigation();
                }}
            }}

            function nextSample() {{
                if (currentIndex < allSamples.length - 1) {{
                    currentIndex++;
                    renderCurrentSample();
                    updateNavigation();
                }}
            }}

            // Keyboard navigation
            document.addEventListener('keydown', (e) => {{
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

                if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                    e.preventDefault();
                    prevSample();
                }} else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                    e.preventDefault();
                    nextSample();
                }}
            }});
        </script>
    </body>
    </html>
    """
