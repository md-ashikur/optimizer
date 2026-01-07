const stats = [
  { value: '98.47%', label: 'ML Model Accuracy', color: 'text-purple-400' },
  { value: '1,167+', label: 'Websites Analyzed', color: 'text-pink-400' },
  { value: '21+', label: 'Performance Metrics', color: 'text-blue-400' },
];

export default function StatsSection() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-3xl mx-auto">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="p-6 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10"
        >
          <div className={`text-4xl font-bold ${stat.color} mb-2`}>
            {stat.value}
          </div>
          <div className="text-gray-300">{stat.label}</div>
        </div>
      ))}
    </div>
  );
}
