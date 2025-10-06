/**
 * System Status Component for NASA Space Apps Hackathon MVP
 * Shows API health, system statistics, and connection status
 */

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { useSystem } from '@/context/ApiContext';
import { 
  Server, 
  Database, 
  Brain, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  RefreshCw,
  Activity,
  FileText,
  Network
} from 'lucide-react';
import { apiUtils } from '@/services/api';

const SystemStatus: React.FC = () => {
  const { checkHealth, getStats, triggerIngest, healthStatus, stats, isConnected } = useSystem();
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    // Initial health check
    checkHealth();
    getStats();
  }, [checkHealth, getStats]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([checkHealth(), getStats()]);
    setIsRefreshing(false);
  };

  const handleIngest = async () => {
    await triggerIngest();
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      default:
        return <XCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      default:
        return 'bg-red-500';
    }
  };

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Status da Conexão
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isConnected ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <span className="font-medium">
                {isConnected ? 'Conectado' : 'Desconectado'}
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              {isRefreshing ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* API Health */}
      {healthStatus && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="w-5 h-5" />
              Saúde da API
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span>Status Geral</span>
                <div className="flex items-center gap-2">
                  {getStatusIcon(healthStatus.status)}
                  <Badge className={getStatusColor(healthStatus.status)}>
                    {healthStatus.status}
                  </Badge>
                </div>
              </div>
              
              {healthStatus.components && (
                <div className="space-y-2">
                  <span className="text-sm font-medium">Componentes:</span>
                  {Object.entries(healthStatus.components).map(([component, isHealthy]) => (
                    <div key={component} className="flex items-center justify-between text-sm">
                      <span className="capitalize">{component}</span>
                      {isHealthy ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-500" />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Statistics */}
      {stats && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Estatísticas do Sistema
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Vector Store Stats */}
              {stats.vectorstore && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    <span className="font-medium">Vector Store</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>Documentos: {stats.vectorstore.documents || 0}</div>
                    <div>Chunks: {stats.vectorstore.chunks || 0}</div>
                    <div>Embeddings: {stats.vectorstore.embeddings || 0}</div>
                    <div>Dimensões: {stats.vectorstore.dimensions || 'N/A'}</div>
                  </div>
                </div>
              )}

              {/* Knowledge Graph Stats */}
              {stats.knowledge_graph && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Network className="w-4 h-4" />
                    <span className="font-medium">Knowledge Graph</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>Nós: {stats.knowledge_graph.nodes || 0}</div>
                    <div>Relacionamentos: {stats.knowledge_graph.relationships || 0}</div>
                    <div>Entidades: {stats.knowledge_graph.entities || 0}</div>
                    <div>Triplas: {stats.knowledge_graph.triples || 0}</div>
                  </div>
                </div>
              )}

              {/* Mission Engine Stats */}
              {stats.mission_engine && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4" />
                    <span className="font-medium">Mission Engine</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>Consultas: {stats.mission_engine.queries || 0}</div>
                    <div>Respostas: {stats.mission_engine.responses || 0}</div>
                    <div>Perfis: {stats.mission_engine.profiles || 0}</div>
                    <div>Temas: {stats.mission_engine.themes || 0}</div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Actions */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Ações do Sistema</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Button
              onClick={handleIngest}
              className="w-full"
              variant="outline"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Reprocessar Documentos
            </Button>
            <p className="text-xs text-muted-foreground">
              Processa novamente todos os PDFs e reconstrói o conhecimento
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemStatus;
