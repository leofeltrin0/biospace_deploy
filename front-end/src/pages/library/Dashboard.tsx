import { Card } from "@/components/ui/card";
import { academicWorks } from "./Chat";
import { ChevronDown, ChevronUp, Brain, FileText, Network } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { usePaper } from "@/context/PaperContext";
import { useSystem, useGraph } from "@/context/ApiContext";
import SystemStatus from "@/components/SystemStatus";
import './dashboard-scrollbar.css';

interface DashboardProps {
    icon: React.ComponentType<{ className?: string }>;
    text: string;
    color: string;
}

const Dashboard = (props: DashboardProps) => {
    const { paperIndex, setPaperIndex } = usePaper();
    const [openDashboard, setOpenDashboard] = useState<boolean>(false);
    const [showSystemStatus, setShowSystemStatus] = useState<boolean>(false);
    const { stats, isConnected } = useSystem();
    const { graphData } = useGraph();

    function isNotValid(paperIndex:number):boolean {
        return paperIndex === null || isNaN(paperIndex) || paperIndex < 0 || paperIndex >= academicWorks.length
    }

    useEffect(() => {
        setOpenDashboard(!isNotValid(paperIndex));
    },[paperIndex])

    return (
        <div className={`relative w-full overflow-hidden ${openDashboard ? 'pt-8' : 'py-8'} border-y border-border`}>
            <div className="flex px-4 gap-6">
                <Card
                className="glass-card flex-shrink-0 w-72 p-6 flex items-center gap-4 hover:scale-105 transition-transform"
                >
                    <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                        <props.icon className={`w-6 h-6 ${props.color}`} />
                    </div>
                    <div>
                        <span className="text-lg font-semibold text-gray-500">Vamos falar sobre:</span><br />         
                        <span className="text-lg font-semibold">{props.text}</span>
                    </div>
                </Card>
                {isNotValid(paperIndex) ? (
                    <div className="flex w-full items-center justify-center">
                        <p className="text-muted-foreground"><i>Selecione uma publicação para visualizar um painel de dados detalhados.</i></p>
                    </div>
                ) : (
                    <div className="flex w-full items-center justify-between">
                        <div>
                            <p className="text-lg"><b>Trabalho selecionado:</b> {academicWorks[paperIndex]}</p>                            
                            <p className="text-muted-foreground">
                                <a href="#" className="text-primary">Acesse o artigo completo</a> ou expanda para ver detalhes como resumo, insights, e mais.
                                {!isConnected && <span className="text-yellow-500 ml-2">⚠️ API desconectada</span>}
                            </p>
                        </div>
                        <div className="flex gap-2">
                            <Button
                                className="hover:bg-transparent hover:text-white border border-secondary"
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowSystemStatus(!showSystemStatus)}
                            >
                                Sistema
                            </Button>
                            <Button
                                className="hover:bg-transparent hover:text-white border border-primary"
                                variant="ghost"
                                size="icon"
                                onClick={() => setOpenDashboard(!openDashboard)}
                            >
                                {openDashboard ? <ChevronUp /> : <ChevronDown />}
                            </Button>
                        </div>
                    </div>
                )}
            </div>
            {showSystemStatus && (
                <div className="py-8 px-4">
                    <SystemStatus />
                </div>
            )}
            
            {openDashboard && (
                <div
                    className="py-8 px-4 flex-1 overflow-x-auto custom-scrollbar"
                    ref={el => {
                        if (!el) return;
                        // Remove event listener para evitar duplicidade
                        el.onwheel = null;
                        el.addEventListener('wheel', function(e) {
                            if (e.deltaY !== 0) {
                                el.scrollLeft += e.deltaY;
                                e.preventDefault();
                                e.stopPropagation();
                            }
                        }, { passive: false });
                    }}
                >
                    <div className="flex gap-2">
                        <div className="min-w-[400px] min-h-[200px] w-full bg-white/5 rounded-lg p-4 shadow">
                            <p className="mb-2 flex items-center gap-2">
                                <FileText className="w-4 h-4" />
                                <b>Resumo:</b>
                            </p>
                            <p className="text-justify">
                                {isConnected ? 
                                    "Análise detalhada do documento selecionado com insights extraídos através de processamento de linguagem natural e análise semântica." :
                                    "Conecte-se à API para visualizar o resumo do documento."
                                }
                            </p>
                        </div>
                        <div className="min-w-[400px] min-h-[200px] w-full bg-white/5 rounded-lg p-4 shadow">
                            <p className="mb-2 flex items-center gap-2">
                                <Brain className="w-4 h-4" />
                                <b>Insights:</b>
                            </p>
                            <ul className="list-disc list-inside space-y-2 text-sm">
                                {isConnected ? (
                                    <>
                                        <li>Análise de entidades e relacionamentos identificados</li>
                                        <li>Classificação temática automática do conteúdo</li>
                                        <li>Extração de conceitos-chave e descobertas principais</li>
                                        <li>Referências cruzadas com outros documentos</li>
                                    </>
                                ) : (
                                    <li>Conecte-se à API para visualizar insights</li>
                                )}
                            </ul>
                        </div>
                        <div className="min-w-[400px] min-h-[200px] w-full bg-white/5 rounded-lg p-4 shadow">
                            <p className="mb-2 flex items-center gap-2">
                                <Network className="w-4 h-4" />
                                <b>Knowledge Graph:</b>
                            </p>
                            <div className="text-sm space-y-2">
                                {graphData ? (
                                    <>
                                        <div>Nós: {graphData.nodes?.length || 0}</div>
                                        <div>Relacionamentos: {graphData.edges?.length || 0}</div>
                                        <div>Estatísticas: {Object.keys(graphData.statistics || {}).length} métricas</div>
                                    </>
                                ) : (
                                    <div>Carregue o grafo de conhecimento para visualizar relacionamentos</div>
                                )}
                            </div>
                        </div>
                        <div className="min-w-[400px] min-h-[200px] w-full bg-white/5 rounded-lg p-4 shadow">
                            <p className="mb-2"><b>Estatísticas do Sistema:</b></p>
                            <div className="text-sm space-y-1">
                                {stats ? (
                                    <>
                                        <div>Status: {isConnected ? '🟢 Conectado' : '🔴 Desconectado'}</div>
                                        <div>Documentos: {stats.vectorstore?.documents || 0}</div>
                                        <div>Chunks: {stats.vectorstore?.chunks || 0}</div>
                                        <div>Entidades: {stats.knowledge_graph?.entities || 0}</div>
                                    </>
                                ) : (
                                    <div>Carregando estatísticas...</div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Dashboard;