
import '../assets/sprites.css';

const NUM_OBJECTS = 29;
const duplicatedSprites = Array.from({ length: NUM_OBJECTS * 3 }).map((_, i) => (i % NUM_OBJECTS) + 1);
  const duplicatedItems = [...duplicatedSprites, ...duplicatedSprites, ...duplicatedSprites];

const InfiniteCarousel = () => {
  return (
    <div className="relative w-full overflow-hidden py-8 border-y border-border/30">
      <div className="flex animate-scroll gap-6">
        {duplicatedItems.map((spriteIndex, index) => (
          <div
            key={index}
            className={`${spriteIndex <= 10 ? 'sprite-lg' : 'sprite-sm'} sprite-${spriteIndex} flex-shrink-0 rounded-xl hover:scale-105 transition-transform`}
            style={{
              width: '60px',
              height: '60px',
              backgroundSize: `${spriteIndex <= 10 ? '1365px 819px' : '341px 409px'}`,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default InfiniteCarousel;