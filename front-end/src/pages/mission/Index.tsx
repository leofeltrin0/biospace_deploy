import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { GiAstronautHelmet } from "react-icons/gi";
import AnswerInput from "@/components/AnswerInput";

const today = new Date();
const day = String(today.getDate()).padStart(2, '0');
const month = String(today.getMonth() + 1).padStart(2, '0');
const formattedDate = `${day}/${month}`;

const Index = () => (
  <div className="min-h-screen flex flex-col bg-main">
    <Header />

    <main className="flex flex-1 justify-center container mx-auto px-4 py-12">
      <div className="flex flex-col items-center justify-center gap-12">
        <div className="text-center space-y-4 max-w-3xl">
          <h2 className="text-5xl md:text-6xl font-bold bg-clip-text">
            <GiAstronautHelmet className="mt-[-1.5rem]" style={
              {
                "display": "inline",
              } as React.CSSProperties
            }/> Missão do dia!
          </h2>
          <p className="text-xl text-muted-foreground">
            Dia {formattedDate}: Descubra como bactérias se comportam em microgravidade
          </p>
        </div>

        <AnswerInput />
      </div>
    </main>

    <Footer />
  </div>
);

export default Index;
