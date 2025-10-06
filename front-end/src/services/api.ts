/**
 * API Service for NASA Space Apps Hackathon MVP
 * Front-end integration with the Space Mission Knowledge Engine back-end
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Types for API responses
export interface QueryRequest {
  query: string;
  user_profile?: 'scientist' | 'manager' | 'layperson';
  max_results?: number;
}

export interface Reference {
  file: string;
  authors: string;
  date: string;
  relevance_score: number;
}

export interface StructuredResponse {
  answer: string;
  references: Reference[];
  theme: string;
  confidence: number;
  key_findings: string[];
  user_profile?: string;
  chunks_used?: number;
  generation_method?: string;
  processing_timestamp?: string;
}

export interface GraphNode {
  id: string;
  label: string;
  group: string;
  size: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  confidence: number;
  weight: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  statistics: Record<string, any>;
}

export interface HealthResponse {
  status: string;
  components: Record<string, boolean>;
  timestamp: string;
}

export interface StatsResponse {
  api: string;
  timestamp: string;
  vectorstore?: Record<string, any>;
  knowledge_graph?: Record<string, any>;
  mission_engine?: Record<string, any>;
}

export interface ThemeResponse {
  themes: string[];
  count: number;
}

export interface ProfileResponse {
  available_profiles: string[];
  count: number;
}

// API Service Class
class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const config = { ...defaultOptions, ...options };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Health and Status
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/stats');
  }

  // Query Endpoints
  async queryMissionInsights(params: QueryRequest): Promise<StructuredResponse> {
    const searchParams = new URLSearchParams({
      query: params.query,
      ...(params.user_profile && { user_profile: params.user_profile }),
      ...(params.max_results && { max_results: params.max_results.toString() }),
    });

    return this.request<StructuredResponse>(`/query?${searchParams}`);
  }

  async queryAdaptive(params: QueryRequest): Promise<StructuredResponse> {
    const searchParams = new URLSearchParams({
      query: params.query,
      ...(params.user_profile && { user_profile: params.user_profile }),
      ...(params.max_results && { max_results: params.max_results.toString() }),
    });

    return this.request<StructuredResponse>(`/query-adaptive?${searchParams}`);
  }

  async queryStructured(params: QueryRequest): Promise<StructuredResponse> {
    const searchParams = new URLSearchParams({
      query: params.query,
      ...(params.user_profile && { user_profile: params.user_profile }),
      ...(params.max_results && { max_results: params.max_results.toString() }),
    });

    return this.request<StructuredResponse>(`/query-structured?${searchParams}`);
  }

  // Knowledge Graph
  async getGraphData(): Promise<GraphResponse> {
    return this.request<GraphResponse>('/graph');
  }

  async getEntityRelations(entity: string, maxDepth: number = 2): Promise<{
    entity: string;
    relations: any[];
    count: number;
  }> {
    const searchParams = new URLSearchParams({
      max_depth: maxDepth.toString(),
    });

    return this.request(`/graph/entities/${entity}?${searchParams}`);
  }

  async getCentralEntities(topK: number = 10): Promise<{
    central_entities: any[];
    count: number;
  }> {
    const searchParams = new URLSearchParams({
      top_k: topK.toString(),
    });

    return this.request(`/graph/central?${searchParams}`);
  }

  // User Profiles and Themes
  async getUserProfiles(): Promise<ProfileResponse> {
    return this.request<ProfileResponse>('/profiles');
  }

  async detectUserProfile(query: string): Promise<{
    query: string;
    detected_profile: string;
    confidence: string;
  }> {
    return this.request('/profiles/detect', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  async getThemes(): Promise<ThemeResponse> {
    return this.request<ThemeResponse>('/themes');
  }

  async classifyTheme(query: string): Promise<{
    query: string;
    classified_theme: string;
    available_themes: string[];
  }> {
    return this.request('/classify-theme', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  // Document Processing
  async triggerIngest(): Promise<{
    success: boolean;
    message: string;
    processed_files?: number;
    errors?: string[];
  }> {
    return this.request('/ingest', {
      method: 'POST',
    });
  }

  // Prompt Engineering Stats
  async getPromptEngineeringStats(): Promise<{
    reference_extraction: boolean;
    theme_classification: boolean;
    structured_output: boolean;
    available_themes: string[];
  }> {
    return this.request('/prompt-engineering/stats');
  }
}

// Create and export the API service instance
export const apiService = new ApiService();

// Export the class for testing
export { ApiService };

// Utility functions for common operations
export const apiUtils = {
  // Format references for display
  formatReferences: (references: Reference[]): string => {
    if (!references || references.length === 0) {
      return 'No references available';
    }

    return references
      .map((ref, index) => {
        const authors = ref.authors !== 'Not available' ? ref.authors : 'Unknown authors';
        const date = ref.date !== 'Not available' ? ref.date : 'Unknown date';
        return `${index + 1}. ${ref.file} - ${authors} (${date})`;
      })
      .join('\n');
  },

  // Get theme color for UI
  getThemeColor: (theme: string): string => {
    const themeColors: Record<string, string> = {
      biotechnology: 'bg-blue-500',
      neuroscience: 'bg-purple-500',
      biochemistry: 'bg-green-500',
      ecology: 'bg-emerald-500',
      microbiology: 'bg-orange-500',
      genetics: 'bg-pink-500',
    };
    return themeColors[theme] || 'bg-gray-500';
  },

  // Get theme icon
  getThemeIcon: (theme: string): string => {
    const themeIcons: Record<string, string> = {
      biotechnology: '🧬',
      neuroscience: '🧠',
      biochemistry: '⚗️',
      ecology: '🌱',
      microbiology: '🦠',
      genetics: '🧬',
    };
    return themeIcons[theme] || '📚';
  },

  // Format confidence as percentage
  formatConfidence: (confidence: number): string => {
    return `${Math.round(confidence * 100)}%`;
  },

  // Truncate text for display
  truncateText: (text: string, maxLength: number = 200): string => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  },
};
