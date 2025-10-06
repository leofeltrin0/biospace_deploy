/**
 * API Context for managing Space Mission Knowledge Engine API state
 */

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { apiService, StructuredResponse, GraphResponse, HealthResponse, StatsResponse } from '@/services/api';
import { toast } from 'react-toastify';

// Types
interface ApiState {
  isLoading: boolean;
  isConnected: boolean;
  lastResponse: StructuredResponse | null;
  graphData: GraphResponse | null;
  healthStatus: HealthResponse | null;
  stats: StatsResponse | null;
  error: string | null;
}

interface ApiContextType extends ApiState {
  // Query methods
  queryMissionInsights: (query: string, userProfile?: string) => Promise<StructuredResponse | null>;
  queryAdaptive: (query: string, userProfile?: string) => Promise<StructuredResponse | null>;
  queryStructured: (query: string, userProfile?: string) => Promise<StructuredResponse | null>;
  
  // Graph methods
  getGraphData: () => Promise<GraphResponse | null>;
  
  // System methods
  checkHealth: () => Promise<boolean>;
  getStats: () => Promise<StatsResponse | null>;
  triggerIngest: () => Promise<boolean>;
  
  // Utility methods
  clearError: () => void;
  clearLastResponse: () => void;
}

// Create context
const ApiContext = createContext<ApiContextType | undefined>(undefined);

// Provider component
export const ApiProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<ApiState>({
    isLoading: false,
    isConnected: false,
    lastResponse: null,
    graphData: null,
    healthStatus: null,
    stats: null,
    error: null,
  });

  // Helper function to handle API calls with loading and error states
  const handleApiCall = useCallback(async <T,>(
    apiCall: () => Promise<T>,
    successMessage?: string,
    errorMessage?: string
  ): Promise<T | null> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const result = await apiCall();
      
      if (successMessage) {
        toast.success(successMessage);
      }
      
      setState(prev => ({ ...prev, isLoading: false, isConnected: true }));
      return result;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'An unknown error occurred';
      setState(prev => ({ 
        ...prev, 
        isLoading: false, 
        isConnected: false, 
        error: errorMsg 
      }));
      
      if (errorMessage) {
        toast.error(errorMessage);
      } else {
        toast.error(`API Error: ${errorMsg}`);
      }
      
      return null;
    }
  }, []);

  // Query methods
  const queryMissionInsights = useCallback(async (
    query: string, 
    userProfile?: string
  ): Promise<StructuredResponse | null> => {
    const result = await handleApiCall(
      () => apiService.queryMissionInsights({ 
        query, 
        user_profile: userProfile as any 
      }),
      'Query processed successfully!',
      'Failed to process query'
    );
    
    if (result) {
      setState(prev => ({ ...prev, lastResponse: result }));
    }
    
    return result;
  }, [handleApiCall]);

  const queryAdaptive = useCallback(async (
    query: string, 
    userProfile?: string
  ): Promise<StructuredResponse | null> => {
    const result = await handleApiCall(
      () => apiService.queryAdaptive({ 
        query, 
        user_profile: userProfile as any 
      }),
      'Adaptive query processed successfully!',
      'Failed to process adaptive query'
    );
    
    if (result) {
      setState(prev => ({ ...prev, lastResponse: result }));
    }
    
    return result;
  }, [handleApiCall]);

  const queryStructured = useCallback(async (
    query: string, 
    userProfile?: string
  ): Promise<StructuredResponse | null> => {
    const result = await handleApiCall(
      () => apiService.queryStructured({ 
        query, 
        user_profile: userProfile as any 
      }),
      'Structured query processed successfully!',
      'Failed to process structured query'
    );
    
    if (result) {
      setState(prev => ({ ...prev, lastResponse: result }));
    }
    
    return result;
  }, [handleApiCall]);

  // Graph methods
  const getGraphData = useCallback(async (): Promise<GraphResponse | null> => {
    const result = await handleApiCall(
      () => apiService.getGraphData(),
      'Graph data loaded successfully!',
      'Failed to load graph data'
    );
    
    if (result) {
      setState(prev => ({ ...prev, graphData: result }));
    }
    
    return result;
  }, [handleApiCall]);

  // System methods
  const checkHealth = useCallback(async (): Promise<boolean> => {
    const result = await handleApiCall(
      () => apiService.getHealth(),
      undefined,
      'API health check failed'
    );
    
    if (result) {
      setState(prev => ({ 
        ...prev, 
        healthStatus: result,
        isConnected: result.status === 'healthy' || result.status === 'degraded'
      }));
      return result.status === 'healthy' || result.status === 'degraded';
    }
    
    return false;
  }, [handleApiCall]);

  const getStats = useCallback(async (): Promise<StatsResponse | null> => {
    const result = await handleApiCall(
      () => apiService.getStats(),
      'System statistics loaded!',
      'Failed to load system statistics'
    );
    
    if (result) {
      setState(prev => ({ ...prev, stats: result }));
    }
    
    return result;
  }, [handleApiCall]);

  const triggerIngest = useCallback(async (): Promise<boolean> => {
    const result = await handleApiCall(
      () => apiService.triggerIngest(),
      'Document ingestion started! This may take a while...',
      'Failed to start document ingestion'
    );
    
    return result !== null;
  }, [handleApiCall]);

  // Utility methods
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  const clearLastResponse = useCallback(() => {
    setState(prev => ({ ...prev, lastResponse: null }));
  }, []);

  const contextValue: ApiContextType = {
    ...state,
    queryMissionInsights,
    queryAdaptive,
    queryStructured,
    getGraphData,
    checkHealth,
    getStats,
    triggerIngest,
    clearError,
    clearLastResponse,
  };

  return (
    <ApiContext.Provider value={contextValue}>
      {children}
    </ApiContext.Provider>
  );
};

// Hook to use the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

// Custom hooks for specific functionality
export const useQuery = () => {
  const { queryMissionInsights, queryAdaptive, queryStructured, isLoading, lastResponse, error } = useApi();
  
  return {
    queryMissionInsights,
    queryAdaptive,
    queryStructured,
    isLoading,
    lastResponse,
    error,
  };
};

export const useGraph = () => {
  const { getGraphData, graphData, isLoading } = useApi();
  
  return {
    getGraphData,
    graphData,
    isLoading,
  };
};

export const useSystem = () => {
  const { checkHealth, getStats, triggerIngest, healthStatus, stats, isConnected } = useApi();
  
  return {
    checkHealth,
    getStats,
    triggerIngest,
    healthStatus,
    stats,
    isConnected,
  };
};
