/**
 * API Configuration for NASA Space Apps Hackathon MVP
 */

export const API_CONFIG = {
  // Base URL for the Space Mission Knowledge Engine API
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  
  // API Endpoints
  ENDPOINTS: {
    HEALTH: '/health',
    STATS: '/stats',
    QUERY: '/query',
    QUERY_ADAPTIVE: '/query-adaptive',
    QUERY_STRUCTURED: '/query-structured',
    GRAPH: '/graph',
    GRAPH_ENTITIES: '/graph/entities',
    GRAPH_CENTRAL: '/graph/central',
    PROFILES: '/profiles',
    PROFILES_DETECT: '/profiles/detect',
    THEMES: '/themes',
    CLASSIFY_THEME: '/classify-theme',
    INGEST: '/ingest',
    PROMPT_ENGINEERING_STATS: '/prompt-engineering/stats',
  },
  
  // Request Configuration
  REQUEST: {
    TIMEOUT: 30000, // 30 seconds
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000, // 1 second
  },
  
  // Feature Flags
  FEATURES: {
    ENABLE_DEBUG_MODE: import.meta.env.VITE_ENABLE_DEBUG_MODE === 'true',
    ENABLE_API_LOGGING: import.meta.env.VITE_ENABLE_API_LOGGING === 'true',
    ENABLE_STRUCTURED_RESPONSES: true,
    ENABLE_THEME_CLASSIFICATION: true,
    ENABLE_REFERENCE_EXTRACTION: true,
  },
  
  // UI Configuration
  UI: {
    DEFAULT_THEME: import.meta.env.VITE_DEFAULT_THEME || 'dark',
    ENABLE_ANIMATIONS: import.meta.env.VITE_ENABLE_ANIMATIONS !== 'false',
    MAX_RESPONSE_LENGTH: 10000,
    DEBOUNCE_DELAY: 300,
  },
  
  // User Profiles
  USER_PROFILES: [
    { value: 'scientist', label: '🔬 Scientist/Researcher', description: 'Technical and detailed responses' },
    { value: 'manager', label: '👔 Manager/Administrator', description: 'Executive summaries and strategic insights' },
    { value: 'layperson', label: '👤 General Public', description: 'Simple and educational explanations' },
  ],
  
  // Scientific Themes
  THEMES: [
    { value: 'biotechnology', label: '🧬 Biotechnology', color: 'bg-blue-500' },
    { value: 'neuroscience', label: '🧠 Neuroscience', color: 'bg-purple-500' },
    { value: 'biochemistry', label: '⚗️ Biochemistry', color: 'bg-green-500' },
    { value: 'ecology', label: '🌱 Ecology', color: 'bg-emerald-500' },
    { value: 'microbiology', label: '🦠 Microbiology', color: 'bg-orange-500' },
    { value: 'genetics', label: '🧬 Genetics', color: 'bg-pink-500' },
  ],
};

// Helper functions
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

export const getThemeConfig = (theme: string) => {
  return API_CONFIG.THEMES.find(t => t.value === theme) || API_CONFIG.THEMES[0];
};

export const getUserProfileConfig = (profile: string) => {
  return API_CONFIG.USER_PROFILES.find(p => p.value === profile) || API_CONFIG.USER_PROFILES[0];
};

// Debug logging
export const debugLog = (message: string, data?: any) => {
  if (API_CONFIG.FEATURES.ENABLE_DEBUG_MODE) {
    console.log(`[Space Mission Engine] ${message}`, data);
  }
};

export const apiLog = (message: string, data?: any) => {
  if (API_CONFIG.FEATURES.ENABLE_API_LOGGING) {
    console.log(`[API] ${message}`, data);
  }
};
