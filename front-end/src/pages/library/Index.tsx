import Header from "@/components/Header";
import PromptInput from "@/components/PromptInput";
import InfiniteCarousel, { carouselItems } from "@/components/InfiniteCarousel";
import { useSearchParams } from "react-router-dom";
import Footer from "@/components/Footer";
import Chat, { academicWorks } from "./Chat";
import { PaperProvider } from "@/context/PaperContext";
import './dashboard-scrollbar.css';

const Index = () => {

  const [searchParams] = useSearchParams();
  const a = searchParams.get('a');
  const areaIndex = a !== null ? parseInt(a, 10) : null;  
    
  // Se aIndex for válido, exibe o chat e o dashboard sobre os trabalhos
  if (areaIndex !== null && !isNaN(areaIndex) && areaIndex >= 0 && areaIndex < carouselItems.length)
  {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <InfiniteCarousel />
        
        <Chat />
        {/* Linha do Tempo */}
        <section className="border-t border-border/50 glass-card py-6">
          <div className="mx-auto px-4">
            <h3 className="font-semibold mb-4 text-sm text-muted-foreground uppercase tracking-wider">
              Linha do Tempo das Publicações
            </h3>
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
              <div className="flex gap-4 pb-4 relative">
                {academicWorks.map((work, index) => (
                  <div
                    key={index}
                    className=" flex-shrink-0 w-48"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 rounded-full bg-primary shadow-glow flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate ml-[-1.4rem]">(fev/2004)</p>
                        <p className="text-sm font-medium truncate mb-20 ml-[-1.4rem]">{work}</p>
                      </div>
                    </div>
                    {index < academicWorks.length - 1 && (
                      <div className="absolute top-[43%] left-2 w-full h-[2px] bg-gradient-to-r from-primary/50 to-transparent" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
        
        <Footer />
      </div>
    );
  }
  
  // Caso contrário, exibe a home padrão
  return (
    <div className="min-h-screen flex flex-col bg-main">
      <Header />
      <InfiniteCarousel />
      
      <main className="flex-1 container mx-auto px-4 py-12">
        <div className="flex flex-col items-center justify-center gap-12">
          <div className="text-center space-y-4 max-w-3xl">
            <h2 className="text-5xl md:text-6xl font-bold bg-clip-text">
              O que vamos pesquisar hoje?
            </h2>
          </div>
          <PromptInput ativarRecomendacoes={true} />
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
