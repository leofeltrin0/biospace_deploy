import { useSearchParams } from 'react-router-dom';
import { Dna, Microscope, Atom, Leaf, Beaker, Brain } from "lucide-react";
import { Card } from "./ui/card";
import Dashboard from '@/pages/library/Dashboard';

const carouselItems = [
  { icon: Dna, text: "Genética Avançada", color: "text-primary" },
  { icon: Microscope, text: "Microbiologia", color: "text-accent" },
  { icon: Atom, text: "Bioquímica", color: "text-primary" },
  { icon: Leaf, text: "Ecologia", color: "text-accent" },
  { icon: Beaker, text: "Biotecnologia", color: "text-primary" },
  { icon: Brain, text: "Neurociência", color: "text-accent" },
];

const InfiniteCarousel = () => {
  const [searchParams] = useSearchParams();
  const a = searchParams.get('a');
  const areaIndex = a !== null ? parseInt(a, 10) : null;

  // Se aIndex for válido, exibe apenas o card correspondente e remove animação
  if (areaIndex !== null && !isNaN(areaIndex) && areaIndex >= 0 && areaIndex < carouselItems.length) {
    const item = carouselItems[areaIndex];
    const Icon = item.icon;
    return (
      <Dashboard
        icon={Icon}
        text={item.text}
        color={item.color}
      />
    );
  }

  // Caso contrário, exibe o carrossel animado normalmente
  const duplicatedItems = [...carouselItems, ...carouselItems, ...carouselItems];
  return (
    <div className="relative w-full overflow-hidden py-8 border-y border-border" title="Classificação das áreas da biologia espacial">
      <div className="flex animate-scroll gap-6">
        {duplicatedItems.map((item, index) => {
          const Icon = item.icon;
          return (
            <a href={`/library?a=${index%carouselItems.length}`} key={index}>
              <Card className="glass-card flex-shrink-0 w-64 p-6 flex items-center gap-4 hover:scale-105 transition-transform"
              >
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                  <Icon className={`w-6 h-6 ${item.color}`} />
                </div>
                <span className="text-lg font-semibold">{item.text}</span>
              </Card>
            </a>
          );
        })}
      </div>
    </div>
  );
};

export default InfiniteCarousel;
export { carouselItems };