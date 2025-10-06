import { useState } from "react";
import { Send } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { toast } from "react-toastify";

interface AnswerInputProps {
};

const AnswerInput = (props: AnswerInputProps) => {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = () => {
    if (prompt.trim()) {
      console.log("Prompt enviado:", prompt);
      // TODO: Integrar com LLM
      toast.warning("Ops! Investigue mais um pouco, explorador!")
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="glass-card rounded-2xl p-6 glow-primary">
        <div className="flex-1 items-start gap-3 mb-4">
          <h2 className="text-xl font-semibold">Responda corretamente a missão do dia acima</h2>
          <p className="text-sm text-muted-foreground">
            Utilize as funcionalidades do <span className="exo-2-bold">BioSpace Atlas</span> para explorar esta solução
          </p>
        </div>
        
        <div className="relative">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Digite sua resposta para o desafio..."
            className="min-h-[120px] pr-14 bg-background/50 border-border/50 focus:border-primary transition-colors resize-none"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
          />
          <Button
            onClick={handleSubmit}
            size="icon"
            className="absolute bottom-3 right-3 bg-primary hover:bg-primary/90 glow-primary"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AnswerInput;
