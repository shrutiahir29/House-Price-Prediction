import React from "react";
import { Home, Sparkles, Menu, X, BarChart3, Calculator, Info } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, onStartPredicting }) {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const navItems = [
    { id: "home", label: "Home", icon: Home },
    { id: "predict", label: "Predict Price", icon: Calculator },
    { id: "comparison", label: "Model Comparison", icon: BarChart3 },
    { id: "about", label: "About", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-md border-b border-slate-200/60 shadow-xs transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        {/* Logo */}
        <div 
          onClick={() => setActiveTab("home")} 
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-10 h-10 rounded-xl bg-slate-900 text-white flex items-center justify-center shadow-md group-hover:scale-105 transition-transform">
            <Home className="w-5 h-5 text-sky-400" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-xl font-bold tracking-tight text-slate-900 font-display">HouseWise</span>
            </div>
            <p className="text-[11px] font-medium text-slate-500 tracking-wide uppercase">
              Smarter Decisions. Better Homes.
            </p>
          </div>
        </div>

        {/* Center Pill Navigation */}
        <nav className="hidden md:flex items-center bg-slate-100/80 p-1.5 rounded-full border border-slate-200/80 shadow-inner">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`px-5 py-2 rounded-full text-sm font-semibold transition-all duration-200 ${
                  isActive
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Right CTA */}
        <div className="hidden md:flex items-center gap-3">
          <button
            onClick={onStartPredicting}
            className="flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-semibold text-white bg-slate-900 hover:bg-slate-800 active:scale-95 shadow-md shadow-slate-900/10 transition-all cursor-pointer"
          >
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span>Start Predicting</span>
            <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
          </button>
        </div>

        {/* Mobile Hamburger */}
        <div className="md:hidden flex items-center">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden px-4 pt-2 pb-6 bg-white border-b border-slate-200 shadow-xl flex flex-col gap-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setActiveTab(item.id);
                setMobileMenuOpen(false);
              }}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl text-left text-sm font-semibold ${
                activeTab === item.id
                  ? "bg-slate-100 text-slate-900"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <item.icon className="w-4 h-4" />
              <span>{item.label}</span>
            </button>
          ))}
          <button
            onClick={() => {
              onStartPredicting();
              setMobileMenuOpen(false);
            }}
            className="mt-2 w-full flex items-center justify-center gap-2 px-5 py-3 rounded-full text-sm font-semibold text-white bg-slate-900"
          >
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span>Start Predicting →</span>
          </button>
        </div>
      )}
    </header>
  );
}
