import { useState, useEffect } from "react";
import { Send, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Card } from "@/components/ui/card";
import { VscFilePdf } from "react-icons/vsc";
import { usePaper } from "@/context/PaperContext";
import { useQuery } from "@/context/ApiContext";
import { toast } from "react-toastify";
import StructuredResponse from "@/components/StructuredResponse";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  structuredResponse?: any;
};

const academicWorks = [
"Fotossíntese e Respiração Celular",
"Genética Mendeliana",
"Evolução das Espécies",
"Ecossistemas Marinhos",
"Biotecnologia Moderna",
];

const Chat = () => {
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [prompt, setPrompt] = useState("");
  const [userProfile, setUserProfile] = useState<string>("scientist");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Olá! Como posso te ajudar hoje?",
      timestamp: new Date(),
    },
  ]);
  
  const { queryStructured, isLoading, lastResponse } = useQuery();

  const savedNotebooks = [
    "Estudo sobre DNA",
    "Células Tronco",
    "Biodiversidade Brasileira",
    "Proteínas e Enzimas",
  ];

  const { setPaperIndex } = usePaper();

  const handleSubmit = async () => {
    if (prompt.trim()) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: prompt,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, userMessage]);
      setPrompt("");
      
      try {
        // Query the API
        const response = await queryStructured(prompt, userProfile);
        
        if (response) {
          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: response.answer,
            timestamp: new Date(),
            structuredResponse: response,
          };
          
          setMessages(prev => [...prev, assistantMessage]);
          toast.success("Resposta gerada com sucesso!");
        }
      } catch (error) {
        toast.error("Erro ao processar sua pergunta. Tente novamente.");
      }
    }
  };

  return (
    <main className="flex-1 flex">
        {/* Painel Esquerdo - Trabalhos Acadêmicos */}
        <Collapsible
            open={leftPanelOpen}
            onOpenChange={setLeftPanelOpen}
            className="relative border-r border-border/50"
        >
            <div className={`h-full transition-all duration-300 ${leftPanelOpen ? "w-72" : "w-0"}`}>
            <CollapsibleContent className="h-full">
                <div className="h-full glass-card p-4">
                <h3 className="font-semibold text-lg">Publicações da área</h3>
                <p className="text-muted-foreground mb-4">Trabalhos relacionados à categoria selecionada acima</p>
                <ScrollArea className="h-[calc(100vh-200px)]">
                    <div className="space-y-3">
                        <RadioGroup>
                            {academicWorks.map((work, index) => (
                            <div onClick={() => setPaperIndex(index)} key={index} className="flex items-start gap-2">
                                <RadioGroupItem value={`work-${index}`} id={`work-${index}`} />
                                <label
                                    htmlFor={`work-${index}`}
                                    className="text-sm cursor-pointer hover:text-primary transition-colors"
                                >
                                    {work}
                                </label>
                            </div>
                            ))}
                        </RadioGroup>
                    </div>
                </ScrollArea>
                </div>
            </CollapsibleContent>
            </div>
            <CollapsibleTrigger asChild>
            <Button
                variant="ghost"
                size="icon"
                className="absolute top-4 -right-10 z-10"
            >
                {leftPanelOpen ? <ChevronLeft /> : <ChevronRight />}
            </Button>
            </CollapsibleTrigger>
        </Collapsible>

        {/* Centro - Histórico de Conversas */}
        <div className="flex-1 flex flex-col">
            <ScrollArea className="flex-1 p-6">
            <div className="max-w-3xl mx-auto space-y-4">
                {messages.map((message) => (
                <div
                    key={message.id}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                    {message.role === "user" ? (
                        <div className="max-w-[80%] rounded-2xl p-4 bg-primary text-primary-foreground">
                            <p className="text-sm">{message.content}</p>
                            <span className="text-xs opacity-70 mt-2 block">
                                {message.timestamp.toLocaleTimeString()}
                            </span>
                        </div>
                    ) : (
                        <div className="max-w-[80%] w-full">
                            <div className="glass-card rounded-2xl p-4 mb-2">
                                <p className="text-sm">{message.content}</p>
                                <span className="text-xs opacity-70 mt-2 block">
                                    {message.timestamp.toLocaleTimeString()}
                                </span>
                            </div>
                            {message.structuredResponse && (
                                <StructuredResponse 
                                    response={message.structuredResponse} 
                                    className="mt-2"
                                />
                            )}
                        </div>
                    )}
                </div>
                ))}
            </div>
            </ScrollArea>

            {/* Caixa de Texto */}
            <div className="border-t border-border/50 p-4">
                <div className="max-w-3xl mx-auto space-y-3">
                    {/* User Profile Selector */}
                    <div className="flex items-center gap-3">
                        <label className="text-sm font-medium">Perfil:</label>
                        <Select value={userProfile} onValueChange={setUserProfile}>
                            <SelectTrigger className="w-48">
                                <SelectValue placeholder="Selecione seu perfil" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="scientist">🔬 Cientista/Pesquisador</SelectItem>
                                <SelectItem value="manager">👔 Gerente/Administrador</SelectItem>
                                <SelectItem value="layperson">👤 Público Geral</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    
                    <div className="relative">
                        <Textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Digite sua mensagem..."
                            className="min-h-[80px] pr-14 bg-background/50 border-border/50 focus:border-primary transition-colors resize-none"
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSubmit();
                                }
                            }}
                        />
                        <Button
                            onClick={handleSubmit}
                            disabled={isLoading || !prompt.trim()}
                            size="icon"
                            className="absolute bottom-3 right-3 bg-primary hover:bg-primary/90"
                        >
                            {isLoading ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                                <Send className="h-4 w-4" />
                            )}
                        </Button>
                    </div>
                </div>
                <div className="max-w-3xl mx-auto relative mt-4">
                    <div className="flex-1">
                        <p className="text-sm mb-2">
                            Recomendações baseadas em suas últimas pesquisas:
                        </p>
                        <div className="gap-2 flex flex-wrap">
                            <a href="#">
                            <Card
                                className="flex-shrink-0 p-2 flex items-center gap-2 bg-white/10 hover:scale-105 transition-transform"
                            >
                                <div className="w-6 h-6 flex items-center justify-center">
                                <VscFilePdf className={`w-5 h-5 text-red-500`} />
                                </div>
                                <span className="text-sm font-semibold">Trabalho XPTO</span>
                            </Card>
                            </a>

                            <a href="#">
                            <Card
                                className="flex-shrink-0 p-2 flex items-center gap-2 bg-white/10 hover:scale-105 transition-transform"
                            >
                                <div className="w-6 h-6 flex items-center justify-center">
                                <VscFilePdf className={`w-5 h-5 text-red-500`} />
                                </div>
                                <span className="text-sm font-semibold">Trabalho Lorem Ipsum</span>
                            </Card>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        {/* Painel Direito - Cadernos Salvos */}
        <Collapsible
            open={rightPanelOpen}
            onOpenChange={setRightPanelOpen}
            className="relative border-l border-border/50"
        >
            <CollapsibleTrigger asChild>
            <Button
                variant="ghost"
                size="icon"
                className="absolute top-4 -left-10 z-10"
            >
                {rightPanelOpen ? <ChevronRight /> : <ChevronLeft />}
            </Button>
            </CollapsibleTrigger>
            <div className={`h-full transition-all duration-300 ${rightPanelOpen ? "w-72" : "w-0"}`}>
            <CollapsibleContent className="h-full">
                <div className="h-full glass-card p-4">
                <h3 className="font-semibold text-lg">Meus Cadernos</h3>
                <p className="text-muted-foreground mb-4">Anotações salvas ao longo das conversas dentro desta categoria de trabalhos</p>
                <ScrollArea className="h-[calc(100vh-200px)]">
                    <div className="space-y-2">
                    {savedNotebooks.map((notebook, index) => (
                        <button
                        key={index}
                        className="w-full text-left p-3 rounded-lg glass-card hover:bg-primary/10 transition-colors flex justify-between items-center"
                        >
                        <p className="text-sm font-medium">{notebook}</p>
                        <span>...</span>
                        </button>
                    ))}
                    <button
                    className="w-full text-center p-3 rounded-lg glass-card hover:bg-primary/10 transition-colors"
                    >
                    <p className="text-sm font-medium">+</p>
                    </button>
                    </div>
                </ScrollArea>
                </div>
            </CollapsibleContent>
            </div>
        </Collapsible>
    </main>
  );
};

export default Chat;
export {academicWorks};