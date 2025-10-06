/**
 * Component for displaying structured responses from the Space Mission Knowledge Engine
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { StructuredResponse as StructuredResponseType, apiUtils } from '@/services/api';
import { FileText, Users, Calendar, Star, Lightbulb, Brain } from 'lucide-react';

interface StructuredResponseProps {
  response: StructuredResponseType;
  className?: string;
}

const StructuredResponse: React.FC<StructuredResponseProps> = ({ response, className = '' }) => {
  const themeIcon = apiUtils.getThemeIcon(response.theme);
  const themeColor = apiUtils.getThemeColor(response.theme);
  const confidencePercentage = apiUtils.formatConfidence(response.confidence);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Main Answer */}
      <Card className="glass-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Answer
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className={`${themeColor} text-white`}>
                {themeIcon} {response.theme}
              </Badge>
              <Badge variant="secondary">
                {confidencePercentage} confidence
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {response.answer}
          </p>
        </CardContent>
      </Card>

      {/* Key Findings */}
      {response.key_findings && response.key_findings.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5" />
              Key Findings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {response.key_findings.map((finding, index) => (
                <li key={index} className="flex items-start gap-2 text-sm">
                  <span className="text-primary font-bold">{index + 1}.</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* References */}
      {response.references && response.references.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              References ({response.references.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <div className="space-y-3">
                {response.references.map((ref, index) => (
                  <div key={index} className="border rounded-lg p-3 bg-background/50">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-muted-foreground" />
                        <span className="font-medium text-sm">{ref.file}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {Math.round(ref.relevance_score * 100)}% relevant
                      </Badge>
                    </div>
                    
                    <div className="space-y-1 text-xs text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Users className="w-3 h-3" />
                        <span>{ref.authors}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar className="w-3 h-3" />
                        <span>{ref.date}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* Metadata */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-sm">Response Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground">
            <div>
              <span className="font-medium">User Profile:</span>
              <br />
              {response.user_profile === 'scientist' && '🔬 Scientist/Researcher'}
              {response.user_profile === 'manager' && '👔 Manager/Administrator'}
              {response.user_profile === 'layperson' && '👤 General Public'}
            </div>
            <div>
              <span className="font-medium">Generation Method:</span>
              <br />
              {response.generation_method || 'N/A'}
            </div>
            <div>
              <span className="font-medium">Chunks Used:</span>
              <br />
              {response.chunks_used || 0}
            </div>
            <div>
              <span className="font-medium">Timestamp:</span>
              <br />
              {response.processing_timestamp ? 
                new Date(response.processing_timestamp).toLocaleString() : 
                'N/A'
              }
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default StructuredResponse;
