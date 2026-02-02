"use client";

import React, { useState } from 'react';
import { AnalysisResult } from '@/types/performance';

interface Props {
  result: AnalysisResult;
}

export default function ExtraFeaturesPanel({ result }: Props) {
  const [plan, setPlan] = useState<string[] | null>(null);

  function makePlan() {
    const items = (result.recommendations ?? []).slice(0, 5).map((r, i) => `${i + 1}. ${r}`);
    setPlan(items.length ? items : ['Run full audit to generate implementation plan.']);
  }

  return (
    <section className="bg-slate-800 rounded-lg p-6 text-white">
      <div className="grid gap-4">
        <button onClick={makePlan} className="bg-violet-600 hover:bg-violet-500 py-2 px-3 rounded">Plan implementation</button>
      </div>

      <div className="mt-4 space-y-3">
        {plan && (
          <div className="mt-3 bg-slate-700 p-3 rounded">
            <h3 className="font-semibold">Implementation Plan</h3>
            <ul className="list-decimal list-inside">
              {plan.map((p, idx) => <li key={idx}>{p}</li>)}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}
