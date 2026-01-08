"use client";

import React, { useState } from 'react';
import { AnalysisResult } from '@/types/performance';

interface Props {
  result: AnalysisResult;
}

export default function ExtraFeaturesPanel({ result }: Props) {
  const [roi, setRoi] = useState<number | null>(null);
  const [codeSnippet, setCodeSnippet] = useState<string | null>(null);
  const [plan, setPlan] = useState<string[] | null>(null);
  const [improvements, setImprovements] = useState<{ LCP: number; FCP: number } | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  const label = result.prediction?.label ?? 'Average';
  const confidence = result.prediction?.confidence ?? 0.6;

  function predictImprovements() {
    const lcp = result.metrics.Largest_contentful_paint_LCP_ms;
    const fcp = result.metrics.First_Contentful_Paint_FCP_ms;
    const baseFactor = label === 'Weak' ? 0.45 : label === 'Average' ? 0.25 : 0.1;
    const factor = Math.max(0.05, Math.min(0.6, baseFactor * (0.5 + confidence)));
    const improved = {
      LCP: Math.max(600, Math.round(lcp * (1 - factor))),
      FCP: Math.max(300, Math.round(fcp * (1 - factor))),
    };
    setImprovements(improved);
  }

  function calculateROI() {
    const baselineRevenue = 10000;
    const lcp = result.metrics.Largest_contentful_paint_LCP_ms;
    const potentialMs = Math.max(0, lcp - 1200);
    const upliftPct = (potentialMs / 100) * 0.6 * (label === 'Weak' ? 1.2 : label === 'Average' ? 1.0 : 0.7);
    const estimated = Math.round(baselineRevenue * (1 + upliftPct / 100));
    setRoi(estimated);
  }

  async function generateCode() {
    const top = result.recommendations?.[0] ?? '';
    let snippet = '';
    if (/image/i.test(top)) {
      snippet = `<!-- Lazy loading using loading=\"lazy\" -->\n<img src=\"/path/to/image.jpg\" loading=\"lazy\" alt=\"...\" />`;
    } else if (/compress|gzip|brotli/i.test(top)) {
      snippet = `# nginx gzip example\nserver {\n  gzip on;\n  gzip_types text/plain application/json text/css application/javascript;\n}`;
    } else if (/defer|script/i.test(top)) {
      snippet = `<script src=\"/js/app.js\" defer></script>`;
    } else {
      snippet = `<!-- Example: add resource hints -->\n<link rel=\"preload\" href=\"/css/main.css\" as=\"style\">`;
    }

    setCodeSnippet(snippet);
    try {
      await navigator.clipboard.writeText(snippet);
      setCopyStatus('Copied to clipboard');
    } catch (e) {
      setCopyStatus('Copy failed — select and copy manually');
    }
    setTimeout(() => setCopyStatus(null), 3000);
  }

  function makePlan() {
    const items = (result.recommendations ?? []).slice(0, 5).map((r, i) => `${i + 1}. ${r}`);
    setPlan(items.length ? items : ['Run full audit to generate implementation plan.']);
  }

  function visualize() {
    if (!improvements) {
      predictImprovements();
      return;
    }
  }

  function benchmark() {
    const competitors = {
      Good: { LCP: 1200, FCP: 800 },
      Average: { LCP: 2200, FCP: 1400 },
      Weak: { LCP: 3800, FCP: 2500 },
    };
    const comp = competitors[label as keyof typeof competitors] ?? competitors.Average;
    setPlan([`Competitor median LCP: ${comp.LCP}ms`, `Competitor median FCP: ${comp.FCP}ms`]);
  }

  return (
    <section className="bg-slate-800 rounded-lg p-6 text-white">
      <h2 className="text-xl font-semibold mb-4">Additional Insights & Actions</h2>

      <div className="grid gap-4 md:grid-cols-3">
        <button onClick={predictImprovements} className="bg-indigo-600 hover:bg-indigo-500 py-2 px-3 rounded">Predict performance improvements</button>
        <button onClick={calculateROI} className="bg-emerald-600 hover:bg-emerald-500 py-2 px-3 rounded">Calculate business ROI</button>
        <button onClick={generateCode} className="bg-yellow-600 hover:bg-yellow-500 py-2 px-3 rounded">Generate executable code</button>
        <button onClick={makePlan} className="bg-violet-600 hover:bg-violet-500 py-2 px-3 rounded">Plan implementation</button>
        <button onClick={visualize} className="bg-sky-600 hover:bg-sky-500 py-2 px-3 rounded">Visualize outcomes</button>
        <button onClick={benchmark} className="bg-rose-600 hover:bg-rose-500 py-2 px-3 rounded">Benchmark competitors</button>
      </div>

      <div className="mt-4 space-y-3">
        {improvements && (
          <div className="p-3 bg-slate-700 rounded">
            <h3 className="font-semibold">Predicted Improvements</h3>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <div>LCP: <strong>{result.metrics.Largest_contentful_paint_LCP_ms}ms</strong> → <strong>{improvements.LCP}ms</strong></div>
              <div>FCP: <strong>{result.metrics.First_Contentful_Paint_FCP_ms}ms</strong> → <strong>{improvements.FCP}ms</strong></div>
            </div>
          </div>
        )}

        {roi !== null && (
          <div className="p-3 bg-slate-700 rounded">Estimated monthly revenue after improvements: <strong>${roi}</strong></div>
        )}

        {codeSnippet && (
          <div className="mt-1 p-3 bg-black rounded text-xs">
            <div className="flex items-start justify-between">
              <pre className="overflow-auto">{codeSnippet}</pre>
            </div>
            <div className="mt-2 text-xs text-gray-300">{copyStatus}</div>
          </div>
        )}

        {plan && (
          <div className="mt-3 bg-slate-700 p-3 rounded">
            <h3 className="font-semibold">Implementation Plan / Benchmark</h3>
            <ul className="list-decimal list-inside">
              {plan.map((p, idx) => <li key={idx}>{p}</li>)}
            </ul>
          </div>
        )}

        {improvements && (
          <div className="mt-2 p-3 bg-slate-700 rounded">
            <h3 className="font-semibold">Visualization (current vs predicted)</h3>
            <div className="mt-2 space-y-2">
              <MetricBar label="LCP" current={result.metrics.Largest_contentful_paint_LCP_ms} predicted={improvements.LCP} />
              <MetricBar label="FCP" current={result.metrics.First_Contentful_Paint_FCP_ms} predicted={improvements.FCP} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function MetricBar({ label, current, predicted }: { label: string; current: number; predicted: number }) {
  const max = Math.max(current, predicted, 4000);
  const currentPct = Math.min(100, Math.round((current / max) * 100));
  const predictedPct = Math.min(100, Math.round((predicted / max) * 100));
  return (
    <div>
      <div className="flex justify-between text-sm mb-1"><span>{label}</span><span>{current}ms → {predicted}ms</span></div>
      <div className="h-3 bg-slate-600 rounded overflow-hidden">
        <div className="h-3 bg-rose-500" style={{ width: `${currentPct}%` }} />
        <div className="h-3 bg-emerald-400 relative -mt-3" style={{ width: `${predictedPct}%`, opacity: 0.7 }} />
      </div>
    </div>
  );
}
