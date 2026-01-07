# WebOptimizer AI - Component Architecture

## 📁 Project Structure

```
src/
├── app/                          # Next.js app router
│   ├── page.tsx                  # Homepage (refactored)
│   ├── dashboard/
│   │   └── page.tsx             # Dashboard (refactored)
│   └── api/
│       └── analyze/
│           └── route.ts         # Analysis API endpoint
│
├── components/                   # Reusable React components
│   ├── home/                     # Homepage components
│   │   ├── HeroSection.tsx      # Hero header with badge
│   │   ├── HeroInput.tsx        # URL input form with state
│   │   ├── StatsSection.tsx     # Stats grid (98.47% accuracy, etc.)
│   │   ├── FeaturesSection.tsx  # Features grid with icons
│   │   └── HowItWorksSection.tsx # Step-by-step guide
│   │
│   ├── dashboard/                # Dashboard components
│   │   ├── LoadingState.tsx     # Loading spinner with animation
│   │   ├── ErrorState.tsx       # Error display with retry
│   │   ├── PerformanceGrade.tsx # Grade card with confidence meter
│   │   ├── CoreWebVitals.tsx    # Metrics grid container
│   │   ├── MetricCard.tsx       # Individual metric display
│   │   ├── IssuesList.tsx       # Issues with severity badges
│   │   └── RecommendationsList.tsx # Recommendation cards
│   │
│   └── shared/                   # Shared components
│       ├── Header.tsx            # Navigation header
│       └── Footer.tsx            # Footer with copyright
│
├── store/                        # Zustand state management
│   └── analysis.store.ts        # Analysis state (URL, result, loading, error)
│
├── lib/                          # Utility functions and APIs
│   ├── api/
│   │   └── analysis.api.ts      # API client for website analysis
│   └── utils/
│       ├── theme.utils.ts       # Theme config (colors, badges)
│       └── metrics.utils.ts     # Metric calculations and formatting
│
└── types/                        # TypeScript types
    └── performance.ts            # Interfaces for metrics, predictions, results
```

## 🎨 Key Design Decisions

### 1. **Component Separation**
- Each component has a single responsibility
- Logic is contained within components (no prop drilling)
- Reusable components in `shared/`

### 2. **State Management (Zustand)**
```typescript
// Global state for analysis flow
- currentUrl: string | null
- analysisResult: AnalysisResult | null
- isAnalyzing: boolean
- error: string | null
```

### 3. **Icon Library**
- Using **react-icons** (not lucide-react)
- Import from specific packages: `react-icons/hi`, `react-icons/hi2`
- Example: `import { HiLightningBolt } from 'react-icons/hi'`

### 4. **Utility Functions**
- **Theme utils**: Centralized color/badge configuration
- **Metrics utils**: Status calculation, formatting
- **API utils**: Clean API client layer

## 🔄 Data Flow

```
Homepage → HeroInput → Zustand Store → Dashboard → API → ML Backend
```

1. User enters URL in `HeroInput`
2. URL saved to Zustand store
3. Navigate to dashboard
4. Dashboard reads URL from store
5. Calls API via `analysis.api.ts`
6. Displays loading state
7. Shows results in components

## 🧩 Component Usage

### Homepage
```tsx
<Header />
<HeroSection />
<HeroInput />          // Form with Zustand integration
<StatsSection />
<FeaturesSection />
<HowItWorksSection />
<Footer />
```

### Dashboard
```tsx
<Header />
<PerformanceGrade />   // Grade with confidence meter
<CoreWebVitals />      // Grid of MetricCards
<IssuesList />         // Issues with severity
<RecommendationsList /> // Action items
```

## 📦 Dependencies

- **zustand**: State management
- **react-icons**: Icon library
- **Next.js**: Framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling

## 🚀 Running the Application

```bash
# Install dependencies
npm install

# Start Next.js dev server
npm run dev

# Start Python ML server (separate terminal)
python src/api/ml_server.py
```

## 📊 Best Practices

### Component Design
✅ Single responsibility principle
✅ Props typed with TypeScript
✅ Self-contained logic (no unnecessary prop passing)
✅ Reusable and composable

### State Management
✅ Zustand for global state
✅ Local useState for component-specific state
✅ No prop drilling

### Code Quality
✅ No unnecessary code
✅ Optimized imports
✅ Professional naming conventions
✅ Clean file organization

## 🔧 Configuration Files

- `tsconfig.json`: TypeScript config with path aliases (`@/`)
- `tailwind.config.ts`: Tailwind customization
- `next.config.ts`: Next.js configuration

## 📝 Type Definitions

All types centralized in `src/types/performance.ts`:
- `PerformanceMetrics`: 21+ metric fields
- `PerformanceGrade`: 'Good' | 'Average' | 'Weak'
- `PredictionResult`: ML prediction with confidence
- `AnalysisResult`: Complete analysis response
- `PerformanceIssue`: Issue with severity

## 🎯 Next Steps

1. Test the complete flow
2. Add error boundaries
3. Implement loading skeletons
4. Add animations with Framer Motion (if needed)
5. Optimize performance with React.memo (if needed)
