import { useState } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { usePaper } from "@/context/PaperContext";
import { Checkbox } from "@/components/ui/checkbox";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

const areas = [
"Genética Avançada",
"Microbiologia",
"Bioquímica",
"Ecologia",
"Biotecnologia",
"Neurociência",
];

const academicWorks = [
"Fotossíntese e Respiração Celular",
"Genética Mendeliana",
"Evolução das Espécies",
"Ecossistemas Marinhos",
"Biotecnologia Moderna",
];

const ChatExploration = () => {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Começe selecionando os filtros ao lado para criação de um questionário",
      timestamp: new Date(),
    },
  ]);

  const { setPaperIndex } = usePaper();

  const handleSubmit = () => {
    if (prompt.trim()) {
      const newMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: prompt,
        timestamp: new Date(),
      };
      setMessages([...messages, newMessage]);
      setPrompt("");
      // TODO: Integrar com LLM
    }
  };

  return (
    <main className="flex-1 flex">
        {/* Painel 1 - Áreas dos trabalhos */}
        <div className={`h-full border-r border-border/50 transition-all duration-300 w-72`}>
            <div className="h-full glass-card p-4">
                <h3 className="font-semibold text-lg">Classificação em áreas do conhecimento</h3>
                <p className="text-muted-foreground mb-4">Quais áreas da biologia espacial você deseja estudar hoje?</p>
                <ScrollArea className="h-[calc(100vh-200px)]">
                    <div className="space-y-3">
                        {areas.map((area, index) => (
                        <div onClick={() => setPaperIndex(index)} key={index} className="flex items-start gap-2">
                            <Checkbox id={`area-${index}`} />
                            <label
                                htmlFor={`area-${index}`}
                                className="text-sm cursor-pointer hover:text-primary transition-colors"
                            >
                                {area}
                            </label>
                        </div>
                        ))}
                    </div>
                </ScrollArea>
            </div>
        </div>

        {/* Painel 2 - Trabalhos Acadêmicos */}
        <div className={`h-full border-r border-border/50 transition-all duration-300 w-72`}>
            <div className="h-full glass-card p-4">
                <h3 className="font-semibold text-lg">Publicações das áreas selecionadas</h3>
                <p className="text-muted-foreground mb-4">Quais trabalhos das áreas selecionadas você deseja considerar no seu estudo?</p>
                <ScrollArea className="h-[calc(100vh-200px)]">
                    <div className="space-y-3">
                        {academicWorks.map((work, index) => (
                        <div onClick={() => setPaperIndex(index)} key={index} className="flex items-start gap-2">
                            <Checkbox id={`work-${index}`} />
                            <label
                                htmlFor={`work-${index}`}
                                className="text-sm cursor-pointer hover:text-primary transition-colors"
                            >
                                {work}
                            </label>
                        </div>
                        ))}
                    </div>
                </ScrollArea>
            </div>
        </div>

        {/* Centro - Histórico de Conversas */}
        <div className="flex-1 flex flex-col">
            <ScrollArea className="flex-1 p-6">
            <div className="max-w-3xl mx-auto space-y-4">
                {messages.map((message) => (
                <div
                    key={message.id}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                    <div
                    className={`max-w-[80%] rounded-2xl p-4 ${
                        message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "glass-card"
                    }`}
                    >
                    <p className="text-sm">{message.content}</p>
                    <span className="text-xs opacity-70 mt-2 block">
                        {message.timestamp.toLocaleTimeString()}
                    </span>
                    </div>
                </div>
                ))}
            </div>
            </ScrollArea>

            {/* Caixa de Texto */}
            <div className="border-t border-border/50 p-4">
                <div className="max-w-3xl mx-auto relative">
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
                    size="icon"
                    className="absolute bottom-3 right-3 bg-primary hover:bg-primary/90"
                    >
                    <Send className="h-4 w-4" />
                    </Button>
                </div>
            </div>
        </div>
    </main>
  );
};

export default ChatExploration;
export {academicWorks};