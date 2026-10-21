import React from "react";
import { Home, Heart } from "lucide-react";

export default function Footer({ onNavigate }) {
  return (
    <footer className="w-full bg-white border-t border-slate-200 mt-16 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-6">
        
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-slate-900 text-white flex items-center justify-center">
            <Home className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-left">
            <span className="text-base font-bold text-slate-900 font-display">HouseWise</span>
            <p className="text-[10px] text-slate-500 font-medium">Smarter Decisions. Better Homes.</p>
          </div>
        </div>

        {/* Links */}
        <div className="flex items-center gap-6 text-xs font-semibold text-slate-600">
          <button onClick={() => onNavigate("home")} className="hover:text-slate-900 transition-colors cursor-pointer">Home</button>
          <button onClick={() => onNavigate("predict")} className="hover:text-slate-900 transition-colors cursor-pointer">Predict Price</button>
          <button onClick={() => onNavigate("comparison")} className="hover:text-slate-900 transition-colors cursor-pointer">Model Comparison</button>
          <button onClick={() => onNavigate("about")} className="hover:text-slate-900 transition-colors cursor-pointer">About Methodology</button>
        </div>

        {/* Copyright */}
        <div className="text-xs text-slate-500">
          Built with React, Vite, FastAPI & Machine Learning
        </div>

      </div>
    </footer>
  );
}
