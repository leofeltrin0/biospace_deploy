import { Card } from '@/components/ui/card';
import { usePaper } from '@/context/PaperContext';
import '@/pages/library/dashboard-scrollbar.css';
import { useEffect, useState } from 'react';
import { FaBook } from 'react-icons/fa6';

interface StoriesBarProps {
  papers: string[]
}

const StoriesBar = (props:StoriesBarProps) => {
  const duplicatedItems = [...props.papers];
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const {setPaperIndex} = usePaper();

  useEffect(() => {
    setPaperIndex(selectedIndex);
  }, [])
  
  return (
    <div 
      className="relative w-full overflow-x-auto p-8 pt-4 border-y border-border custom-scrollbar bg-background"      
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
      <h2 className='mb-4 font-semibold text-muted-foreground/75'>Aqui você consulta todo seu aprendizado desbloqueado durante as pesquisas</h2>
    </div>
  );
};

export default StoriesBar;