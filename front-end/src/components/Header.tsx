import { Leaf, Menu } from "lucide-react";
import { Button } from "./ui/button";
import logo from "@/assets/biospace-logo.png";
import { toast, ToastContainer } from "react-toastify";

const Header = () => {
  const fullUrl = window.location.href;
  return (
    <header className="w-full border-b border-border/50 glass-card">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <a href="/" className="flex items-center gap-3">
            <img src={logo} alt="BioSpace Atlas" className="w-10 h-10" />
            <h1 className="text-2xl font-bold exo-2-regular">
              BioSpace Atlas
            </h1>
          </a>
          
          <nav className="hidden md:flex items-center gap-6">
            <a href="/library" className={`text-foreground/80 hover:text-primary transition-colors ${fullUrl.includes('library') ? 'text-primary' : ''}`}>
              Biblioteca
            </a>
            <a href="/stories" className={`text-foreground/80 hover:text-primary transition-colors ${fullUrl.includes('stories') ? 'text-primary' : ''}`}>
              Histórias
            </a>
            <a href="/exploration" className={`text-foreground/80 hover:text-primary transition-colors ${fullUrl.includes('exploration') ? 'text-primary' : ''}`}>
              Diário de bordo
            </a>
          </nav>

          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
