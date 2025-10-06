import { createContext, useContext, useState, ReactNode } from "react";

type PaperContextType = {
  paperIndex: number | null;
  setPaperIndex: (index: number) => void;
};

const PaperContext = createContext<PaperContextType | undefined>(undefined);

export const PaperProvider = ({ children }: { children: ReactNode }) => {
  const [paperIndex, setPaperIndex] = useState<number | null>(null);
  return (
    <PaperContext.Provider value={{ paperIndex, setPaperIndex }}>
      {children}
    </PaperContext.Provider>
  );
};

export const usePaper = () => {
  const context = useContext(PaperContext);
  if (!context) throw new Error("usePaper deve ser usado dentro do PaperProvider");
  return context;
};