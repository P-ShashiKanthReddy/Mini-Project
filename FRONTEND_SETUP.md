# Neuromorphic YOLOv5 Frontend

Modern web interface for real-time neuromorphic object detection with CNN/SNN comparison.

## Features

- Real-time webcam detection with live video feed
- CNN/SNN mode switching during detection
- Real-time metrics visualization
- Performance comparison charts
- Metrics persistence with Supabase
- Responsive design for all devices

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and builds
- **Supabase** for real-time data storage
- **Recharts** for data visualization
- **Lucide React** for icons

## Prerequisites

- Node.js 18+ and npm
- Python backend running on port 5000
- Supabase account (configured automatically)

## Installation

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Environment setup:**

   The `.env` file is already configured with Supabase credentials. If needed, update:
   ```
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   VITE_API_BASE_URL=http://localhost:5000
   ```

## Running the Frontend

1. **Start the development server:**
   ```bash
   npm run dev
   ```

2. **Open your browser:**
   ```
   http://localhost:5173
   ```

3. **Ensure Python backend is running:**

   The frontend expects a Python Flask/FastAPI server on port 5000. See backend documentation for setup.

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── DetectionView.tsx       # Video feed and controls
│   │   ├── ModeSelector.tsx        # CNN/SNN mode switcher
│   │   ├── MetricsPanel.tsx        # Real-time metrics display
│   │   ├── MetricsChart.tsx        # Performance charts
│   │   └── ComparisonSummary.tsx   # Comparison statistics
│   ├── services/           # API integration
│   │   └── api.ts                  # Backend API calls
│   ├── lib/                # External libraries
│   │   └── supabase.ts             # Supabase client
│   ├── types/              # TypeScript types
│   │   └── index.ts                # Type definitions
│   ├── App.tsx             # Main application
│   ├── App.css             # Global styles
│   └── main.tsx            # Entry point
├── .env                    # Environment variables
├── package.json            # Dependencies
└── vite.config.ts          # Vite configuration
```

## Usage

### Live Detection View

1. **Select Mode**: Choose between CNN (standard) or SNN (neuromorphic) mode
2. **Start Detection**: Click "Start Detection" to begin webcam capture
3. **View Metrics**: Monitor real-time inference time, power, CPU, and GPU usage
4. **Switch Modes**: Toggle between CNN and SNN during detection to compare
5. **Stop Detection**: Click "Stop Detection" when done

### Analytics View

1. **Performance Charts**: View historical comparison of CNN vs SNN metrics
2. **Toggle Metrics**: Switch between inference time and power consumption charts
3. **Comparison Summary**: See average metrics and percentage differences
4. **Real-time Updates**: Charts update automatically as new data is collected

## API Endpoints

The frontend expects the following backend endpoints:

- `POST /api/detection/start` - Start detection
- `POST /api/detection/stop` - Stop detection
- `POST /api/detection/mode` - Switch mode
- `GET /api/detection/frame` - Get current frame with detections
- `GET /api/detection/status` - Get detection status
- `GET /api/detection/stream` - Video stream endpoint

## Building for Production

```bash
npm run build
```

The production build will be in the `dist/` directory.

## Deployment

### Static Hosting (Netlify, Vercel, etc.)

1. Build the project: `npm run build`
2. Deploy the `dist/` directory
3. Configure environment variables in your hosting platform

### Docker

```bash
docker build -t neuromorphic-yolov5-frontend .
docker run -p 5173:5173 neuromorphic-yolov5-frontend
```

## Troubleshooting

**Frontend can't connect to backend:**
- Ensure Python backend is running on port 5000
- Check CORS settings in backend
- Verify `VITE_API_BASE_URL` in `.env`

**Supabase connection issues:**
- Verify credentials in `.env`
- Check browser console for errors
- Ensure Supabase project is active

**Webcam not accessible:**
- Grant browser permissions for camera access
- Ensure backend has camera access
- Check browser compatibility (Chrome/Edge recommended)

**Charts not displaying:**
- Collect some metrics first by running detection
- Check browser console for errors
- Verify Supabase table exists

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Adding New Components

1. Create component in `src/components/`
2. Import and use in `App.tsx` or other components
3. Add types to `src/types/index.ts` if needed

### Modifying Styles

- Global styles: `src/App.css`
- CSS variables: Defined in `:root` selector
- Responsive breakpoints: `1200px` and `768px`

## Performance Optimization

- Lazy load components for faster initial load
- Optimize frame polling rate based on network
- Implement virtual scrolling for large datasets
- Use React.memo for expensive components

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of Neuromorphic YOLOv5. See main repository for license details.
