const steps = [
  {
    number: 1,
    title: 'Enter Your URL',
    description: 'Simply paste your website URL and click analyze.',
  },
  {
    number: 2,
    title: 'AI Analysis',
    description: 'Our LightGBM model analyzes 21+ performance metrics in real-time.',
  },
  {
    number: 3,
    title: 'Get Insights',
    description: 'Receive detailed recommendations and actionable steps to improve performance.',
  },
];

export default function HowItWorksSection() {
  return (
    <div id="how-it-works" className="mt-32 max-w-4xl mx-auto">
      <h2 className="text-4xl font-bold text-white text-center mb-16">
        How It Works
      </h2>

      <div className="space-y-8">
        {steps.map((step) => (
          <div key={step.number} className="flex gap-6 items-start">
            <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold">
              {step.number}
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white mb-2">
                {step.title}
              </h3>
              <p className="text-gray-400">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
