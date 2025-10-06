import { BookOpen } from "lucide-react";
import { Card } from "./ui/card";
import { GiAstronautHelmet } from 'react-icons/gi';
import { TbFileUpload } from 'react-icons/tb';
import { HiOutlineRocketLaunch } from 'react-icons/hi2';
import { toast } from 'react-toastify';
import { useNavigate } from "react-router-dom";

const today = new Date();
const day = String(today.getDate()).padStart(2, '0');
const month = String(today.getMonth() + 1).padStart(2, '0');
const formattedDate = `${day}/${month}`;
const actions = [
  {
    title: "Missão do dia!",
    description: `Dia ${formattedDate}: Descubra como bactérias se comportam em microgravidade`,
    icon: GiAstronautHelmet,
    iconClass: "text-primary",
    bgClass: "bg-secondary/10 group-hover:bg-secondary/20",
  },
  {
    title: "Modo história",
    description: "Aprenda sobre um novo experimento através de narrativas envolventes, contadas como uma missão",
    icon: BookOpen,
    iconClass: "text-accent",
    bgClass: "bg-accent/10 group-hover:bg-accent/20",
  },
  {
    title: "Criar módulo de exploração",
    description: "Crie questionários educacionais personalizados!",
    icon: HiOutlineRocketLaunch,
    iconClass: "text-primary",
    bgClass: "bg-primary/10 group-hover:bg-primary/20",
  },
  {
    title: "Compartilhe seu trabalho conosco",
    description: "Publique seu artigo sobre biologia espacial e contribua com a comunidade BioSpace Atlas",
    icon: TbFileUpload,
    iconClass: "text-accent",
    bgClass: "bg-accent/10 group-hover:bg-accent/20",
  },
];

const ActionCards = () => {
  const navigate = useNavigate();

  function handleActionClick(index: number) {
    switch(index) {
      case 0:
        navigate('/mission');
        break;
      case 1:
        navigate('/stories');
        break;
      case 2:
        navigate('/exploration');
        break;
      case 3:
        toast.success("Obrigado! Trabalho recebido com sucesso!");
        break;
      default:
        break;
    }
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl mx-auto">
      {actions.map((action, index) => {
        const Icon = action.icon;
        return (
          <Card
            onClick={() => handleActionClick(index)}
            key={index}
            className="glass-card p-6 hover:scale-105 transition-all duration-300 cursor-pointer group border-border/50 hover:border-primary/50"
          >
            <div className={`w-12 h-12 rounded-xl ${action.bgClass} flex items-center justify-center mb-4 transition-colors`}>
              <Icon className={`w-6 h-6 ${action.iconClass}`} />
            </div>
            <h3 className={`text-lg font-semibold mb-2 group-hover:${action.iconClass} transition-colors`}>
              {action.title}
            </h3>
            <p className="text-sm text-muted-foreground">
              {action.description}
            </p>
          </Card>
        );
      })}
    </div>
  );
};

export default ActionCards;
