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
      <h2 className='mb-4 font-semibold text-muted-foreground/75'>Estante de Experimentos</h2>
      <div className="flex gap-6">
        {duplicatedItems.map((item, index) => {
          return (
            <Card onClick={() => {
              setSelectedIndex(index);
              setPaperIndex(index);
            }} className={`glass-card cursor-pointer flex-shrink-0 w-72 p-3 flex items-center gap-4 hover:scale-105 transition-transform ${selectedIndex === index ? 'selected-book' : ''}`}
            >
              <div className="w-12 h-12 rounded-xl bg-[#FFA500]/10 flex items-center justify-center">
                <FaBook className={`w-6 h-6 text-[#FFA500]`} />
              </div>
              <span>{item}</span>
            </Card>
          );
        })}
      </div>
    </div>
  );
};

export default StoriesBar;