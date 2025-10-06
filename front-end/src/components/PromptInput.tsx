import { useState } from "react";
import { Send, Sparkles, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Card } from "./ui/card";
import { VscFilePdf } from "react-icons/vsc";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@/context/ApiContext";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

interface PromptInputProps {
  enableRecommendations?: boolean,
};

const PromptInput = (props: PromptInputProps) => {
  const [prompt, setPrompt] = useState("");
  const [userProfile, setUserProfile] = useState<string>("scientist");
  const navigate = useNavigate();
  const { queryStructured, isLoading, lastResponse } = useQuery();

  const handleSubmit = async () => {
    if (prompt.trim()) {
      console.log("Prompt sent:", prompt);
      
      // Query the API with structured response
      const response = await queryStructured(prompt, userProfile);
      
      if (response) {
        console.log("API Response:", response);
        // Navigate to appropriate page based on user profile
        if (userProfile === "layperson") {
          navigate('/stories');
        } else {
          navigate('/library');
        }
      }
      
      setPrompt("");      
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="glass-card rounded-2xl p-6 glow-primary">
        <div className="flex items-start gap-3 mb-4">
          <Sparkles className="w-6 h-6 text-accent flex-shrink-0 mt-1" />
          <div className="flex-1">
            <h2 className="text-xl font-semibold">If you want to <span className="text-accent">research or learn</span></h2>
            <p className="">
              Ask Atlas now!
            </p>
          </div>
        </div>

        {/* User Profile Selector */}
        <div className="mb-4">
          <label className="text-sm font-medium mb-2 block">Select your profile:</label>
          <Select value={userProfile} onValueChange={setUserProfile}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select your profile" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="scientist">🔬 Scientist/Researcher</SelectItem>
              <SelectItem value="manager">👔 Manager/Administrator</SelectItem>
              <SelectItem value="layperson">👤 General Public</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div className="relative">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type your question about space biology, ecosystems, evolution, mutation..."
            className="min-h-[240px] pr-14 bg-background/50 border-border/50 focus:border-primary transition-colors resize-none"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
          />
          <div className="absolute bottom-3 right-3">          
            <Button
              onClick={handleSubmit}
              disabled={isLoading || !prompt.trim()}
              className="bg-accent hover:bg-accent/90 glow-accent"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4 mr-2" />
                  Send
                </>
              )}
            </Button>
          </div>
        </div>

        {props.enableRecommendations && (
          <div className="flex items-start gap-3 mb-4 mt-4">
            <div className="flex-1">
              <p className="text-sm mb-2">
                Recommendations based on your recent research:
              </p>
              <div className="gap-2 flex flex-wrap">
                <a href="#">
                  <Card
                    className="flex-shrink-0 p-2 flex items-center gap-2 bg-white/10 hover:scale-105 transition-transform"
                  >
                    <div className="w-6 h-6 flex items-center justify-center">
                      <VscFilePdf className={`w-5 h-5 text-red-500`} />
                    </div>
                    <span className="text-sm font-semibold">Research Paper A</span>
                  </Card>
                </a>

                <a href="#">
                  <Card
                    className="flex-shrink-0 p-2 flex items-center gap-2 bg-white/10 hover:scale-105 transition-transform"
                  >
                    <div className="w-6 h-6 flex items-center justify-center">
                      <VscFilePdf className={`w-5 h-5 text-red-500`} />
                    </div>
                    <span className="text-sm font-semibold">Research Paper B</span>
                  </Card>
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PromptInput;
