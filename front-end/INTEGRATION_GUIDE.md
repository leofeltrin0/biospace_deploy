# NASA Space Apps Hackathon MVP - Front-end Integration Guide

## Overview

This guide explains how the front-end React application integrates with the Space Mission Knowledge Engine back-end API. The integration provides a seamless user experience for querying space biology research documents with structured responses, theme classification, and reference extraction.

## Architecture

### Front-end Stack
- **React 18** with TypeScript
- **Vite** for build tooling
- **Tailwind CSS** for styling
- **Shadcn/ui** for UI components
- **React Router** for navigation
- **TanStack Query** for data fetching
- **React Toastify** for notifications

### Back-end Integration
- **FastAPI** REST API
- **Structured JSON responses** with references and themes
- **Real-time health monitoring**
- **System statistics and metrics**

## Key Features

### 1. API Integration Layer
- **`src/services/api.ts`**: Complete API service with TypeScript types
- **`src/context/ApiContext.tsx`**: React context for API state management
- **`src/config/api.ts`**: Configuration and constants

### 2. Structured Response Display
- **`src/components/StructuredResponse.tsx`**: Displays API responses with:
  - Answer text
  - Reference metadata (file, authors, date)
  - Theme classification
  - Confidence scores
  - Key findings

### 3. User Profile Adaptation
- **Scientist**: Technical, detailed responses
- **Manager**: Executive summaries and strategic insights
- **Layperson**: Simple, educational explanations

### 4. System Monitoring
- **`src/components/SystemStatus.tsx`**: Real-time API health monitoring
- Connection status indicators
- System statistics display
- Document processing status

## API Endpoints Integration

### Query Endpoints
```typescript
// Basic query
GET /query?query=space+biology&user_profile=scientist

// Adaptive query with user profile detection
GET /query-adaptive?query=space+biology

// Structured query with references and themes
GET /query-structured?query=space+biology&user_profile=scientist
```

### System Endpoints
```typescript
// Health check
GET /health

// System statistics
GET /stats

// Knowledge graph data
GET /graph

// User profiles
GET /profiles

// Scientific themes
GET /themes
```

## Component Integration

### 1. PromptInput Component
- **Location**: `src/components/PromptInput.tsx`
- **Features**:
  - User profile selection
  - Real-time API integration
  - Loading states
  - Error handling

### 2. Chat Components
- **Library Chat**: `src/pages/library/Chat.tsx`
- **Exploration Chat**: `src/pages/exploration/ChatExploration.tsx`
- **Features**:
  - Message history with structured responses
  - User profile switching
  - Reference display
  - Theme classification

### 3. Dashboard Integration
- **Location**: `src/pages/library/Dashboard.tsx`
- **Features**:
  - System status monitoring
  - Real-time statistics
  - Knowledge graph visualization
  - Document insights

## Configuration

### Environment Variables
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Feature Flags
VITE_ENABLE_DEBUG_MODE=true
VITE_ENABLE_API_LOGGING=true

# UI Configuration
VITE_DEFAULT_THEME=dark
VITE_ENABLE_ANIMATIONS=true
```

### API Configuration
```typescript
// src/config/api.ts
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  ENDPOINTS: {
    QUERY_STRUCTURED: '/query-structured',
    HEALTH: '/health',
    STATS: '/stats',
    // ... other endpoints
  },
  FEATURES: {
    ENABLE_STRUCTURED_RESPONSES: true,
    ENABLE_THEME_CLASSIFICATION: true,
    ENABLE_REFERENCE_EXTRACTION: true,
  }
};
```

## Usage Examples

### 1. Basic Query Integration
```typescript
import { useQuery } from '@/context/ApiContext';

const MyComponent = () => {
  const { queryStructured, isLoading, lastResponse } = useQuery();

  const handleQuery = async (query: string, userProfile: string) => {
    const response = await queryStructured(query, userProfile);
    if (response) {
      console.log('Answer:', response.answer);
      console.log('References:', response.references);
      console.log('Theme:', response.theme);
    }
  };

  return (
    <div>
      {isLoading && <div>Processing...</div>}
      {lastResponse && (
        <StructuredResponse response={lastResponse} />
      )}
    </div>
  );
};
```

### 2. System Monitoring
```typescript
import { useSystem } from '@/context/ApiContext';

const SystemMonitor = () => {
  const { checkHealth, getStats, isConnected, healthStatus } = useSystem();

  useEffect(() => {
    checkHealth();
    getStats();
  }, []);

  return (
    <div>
      <div>Status: {isConnected ? 'Connected' : 'Disconnected'}</div>
      {healthStatus && (
        <div>API Health: {healthStatus.status}</div>
      )}
    </div>
  );
};
```

## Error Handling

### API Error Management
- **Network errors**: Automatic retry with exponential backoff
- **API errors**: User-friendly error messages
- **Timeout handling**: 30-second timeout with fallback
- **Connection monitoring**: Real-time connection status

### User Experience
- **Loading states**: Visual feedback during API calls
- **Error notifications**: Toast messages for errors
- **Graceful degradation**: Fallback content when API unavailable

## Development Workflow

### 1. Start Back-end
```bash
# Terminal 1: Start the API server
cd /path/to/backend
python main.py --mode serve
```

### 2. Start Front-end
```bash
# Terminal 2: Start the React development server
cd front-end
npm run dev
```

### 3. Test Integration
- Navigate to `http://localhost:8080`
- Test query functionality
- Verify API responses
- Check system status

## Deployment Considerations

### Production Configuration
- Update `VITE_API_BASE_URL` to production API URL
- Configure CORS settings in back-end
- Set up proper error monitoring
- Implement rate limiting

### Performance Optimization
- Implement response caching
- Use React Query for data fetching
- Optimize bundle size
- Implement lazy loading

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure back-end CORS is configured for front-end domain
   - Check API base URL configuration

2. **API Connection Issues**
   - Verify back-end is running on correct port
   - Check network connectivity
   - Review API health endpoint

3. **Response Parsing Errors**
   - Validate API response format
   - Check TypeScript type definitions
   - Review error handling logic

### Debug Mode
```typescript
// Enable debug logging
VITE_ENABLE_DEBUG_MODE=true
VITE_ENABLE_API_LOGGING=true
```

## Future Enhancements

### Planned Features
- **Real-time collaboration**: WebSocket integration
- **Advanced visualizations**: D3.js integration for knowledge graphs
- **Offline support**: Service worker implementation
- **Mobile optimization**: Responsive design improvements

### API Extensions
- **Streaming responses**: Real-time response streaming
- **Batch processing**: Multiple query support
- **Advanced filtering**: Complex query parameters
- **Export functionality**: PDF/JSON export options

## Support

For technical support or questions about the integration:
- Review the API documentation
- Check the back-end logs
- Verify configuration settings
- Test with simple queries first

---

**Note**: This integration maintains the back-end logic unchanged while providing a rich front-end experience for the NASA Space Apps Hackathon MVP.
