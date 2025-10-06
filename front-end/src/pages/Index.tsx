import Header from "@/components/Header";
import PromptInput from "@/components/PromptInput";
import ActionCards from "@/components/ActionCards";
import { VscLightbulbSparkle } from 'react-icons/vsc';
import Footer from "@/components/Footer";
import { Send, Sparkles } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { GiAstronautHelmet } from "react-icons/gi";
import { FaAngleRight } from "react-icons/fa6";

const Index = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen flex flex-col bg-main">
      <Header />
      
      <main className="flex-1 container mx-auto px-4 py-12 items-center">
        <div className="flex flex-col items-center justify-center gap-12">
          <div className="text-center space-y-4 max-w-3xl">
            <h2 className="text-5xl md:text-6xl font-bold bg-clip-text">
              Explore the Universe of Space Biology
            </h2>
            <p className="text-xl text-muted-foreground">
              An educational journey through space and life
            </p>
          </div>

          <div className="flex w-full px-10">
            <PromptInput />
            
            <div className="mx-auto cursor-pointer" onClick={() => {navigate('/mission');}}>
              <div className="relative glass-card rounded-2xl p-6 glow-primary">
                <div className="flex items-start gap-3 mb-4">
                  <GiAstronautHelmet className="w-12 h-12 text-primary flex-shrink-0 mt-1" />
                  <div className="flex-1">
                    <h2 className="text-xl font-semibold">If you prefer to <span className="text-primary">challenge yourself</span></h2>
                    <p>
                      Accept our Mission of the day!
                    </p>
                  </div>
                </div>
                <Button className="absolute right-[-1rem] bottom-[-1rem]"><FaAngleRight/></Button>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
