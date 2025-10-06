import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import IndexMission from "./pages/mission/Index";
import IndexStories from "./pages/stories/Index";
import IndexExploration from "./pages/exploration/Index";
import IndexLibrary from "./pages/library/Index";
import NotFound from "./pages/NotFound";
import { ToastContainer } from "react-toastify";
import { PaperProvider } from "./context/PaperContext";
import { ApiProvider } from "./context/ApiContext";

const queryClient = new QueryClient();

const App = () => (
  <ApiProvider>
    <PaperProvider>
      <QueryClientProvider client={queryClient}>    
        <ToastContainer
          position="top-center"
          autoClose={3000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick={false}
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="dark"
        />
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/mission" element={<IndexMission />} />
              <Route path="/stories" element={<IndexStories />} />
              <Route path="/exploration" element={<IndexExploration />} />
              <Route path="/library" element={<IndexLibrary />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </PaperProvider>
  </ApiProvider>
);

export default App;
