import React, { useState } from "react";
import { MapPin, Home as HomeIcon, BedDouble, Maximize2, Sparkles, BarChart2, ArrowRight } from "lucide-react";

export default function HeroSection({ onPredictClick, onExploreModelsClick, onQuickEstimateSubmit }) {
  const [city, setCity] = useState("Pune");
  const [propertyType, setPropertyType] = useState("Villa");
  const [bhk, setBhk] = useState("3");
  const [area, setArea] = useState("1850");

  const handleSubmit = (e) => {
    e.preventDefault();
    onQuickEstimateSubmit({
      city,
      propertyType,
      bhk: parseInt(bhk, 10),
      area: parseFloat(area) || 1200,
    });
  };

  return (
    <section className="relative w-full pt-6 pb-20 md:pb-28 overflow-hidden">
      {/* Hero Visual Card Container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="relative rounded-3xl overflow-hidden min-h-[580px] md:min-h-[640px] shadow-2xl border border-white/60 flex flex-col justify-between p-6 sm:p-10 lg:p-14">
          
          {/* Background Image with Ambient Sunset Overlay */}
          <div className="absolute inset-0 z-0">
            <img
              src="/hero_villa.jpg"
              alt="Luxury Modern Hillside Villa"
              className="w-full h-full object-cover object-center transform scale-105 hover:scale-100 transition-transform duration-1000 ease-out"
            />
            {/* Gradient Overlays: Lightened editorial overlay on left for readable text, preserving sunset warmth */}
            <div className="absolute inset-0 bg-gradient-to-r from-white/95 via-white/80 to-transparent w-full md:w-3/5 lg:w-1/2" />
            <div className="absolute inset-0 bg-gradient-to-t from-slate-900/30 via-transparent to-transparent" />
          </div>

          {/* Left Text Content */}
          <div className="relative z-10 max-w-xl text-left my-auto">
            {/* Kicker */}
            <div className="inline-flex items-center gap-2 mb-4">
              <span className="w-8 h-[2px] bg-amber-600 inline-block" />
              <span className="text-xs sm:text-sm font-bold tracking-widest text-slate-800 uppercase">
                AI POWERED HOUSE PRICE PREDICTION
              </span>
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 leading-[1.1] font-display mb-6">
              Find Your Home's <br />
              <span className="italic font-normal">True Value</span>
            </h1>

            {/* Subtitle */}
            <p className="text-base sm:text-lg text-slate-700 leading-relaxed font-normal mb-8 max-w-lg">
              Get accurate and reliable house price estimates using advanced machine learning models. 
              Make smarter real estate decisions with HouseWise.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center gap-4">
              <button
                onClick={onPredictClick}
                className="flex items-center gap-2 px-7 py-3.5 rounded-full font-semibold text-white bg-slate-900 hover:bg-slate-800 active:scale-95 shadow-lg shadow-slate-900/20 transition-all cursor-pointer group"
              >
                <span>Predict a Price</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>

              <button
                onClick={onExploreModelsClick}
                className="flex items-center gap-2 px-6 py-3.5 rounded-full font-semibold text-slate-800 bg-white/70 hover:bg-white/90 backdrop-blur-md border border-white/60 shadow-md active:scale-95 transition-all cursor-pointer"
              >
                <BarChart2 className="w-4 h-4 text-slate-700" />
                <span>Explore Models</span>
              </button>
            </div>
          </div>

          {/* Floating Glassmorphic Quick Search Bar */}
          <div className="relative z-20 mt-12 w-full">
            <form
              onSubmit={handleSubmit}
              className="glass-hero-card rounded-2xl md:rounded-full p-3 sm:p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 items-center shadow-2xl"
            >
              {/* Field 1: City */}
              <div className="flex items-center gap-3 px-4 py-2.5 rounded-full bg-white/70 hover:bg-white/90 border border-white/60 transition-colors">
                <MapPin className="w-5 h-5 text-slate-700 shrink-0" />
                <div className="text-left w-full">
                  <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    City
                  </label>
                  <select
                    value={city}
                    onChange={(e) => setCity(e.target.value)}
                    className="w-full bg-transparent text-sm font-semibold text-slate-900 focus:outline-none cursor-pointer"
                  >
                    <option value="Pune">Pune</option>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Delhi">Delhi</option>
                    <option value="Hyderabad">Hyderabad</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Ahmedabad">Ahmedabad</option>
                    <option value="Jaipur">Jaipur</option>
                    <option value="Chandigarh">Chandigarh</option>
                  </select>
                </div>
              </div>

              {/* Field 2: Property Type */}
              <div className="flex items-center gap-3 px-4 py-2.5 rounded-full bg-white/70 hover:bg-white/90 border border-white/60 transition-colors">
                <HomeIcon className="w-5 h-5 text-slate-700 shrink-0" />
                <div className="text-left w-full">
                  <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Property Type
                  </label>
                  <select
                    value={propertyType}
                    onChange={(e) => setPropertyType(e.target.value)}
                    className="w-full bg-transparent text-sm font-semibold text-slate-900 focus:outline-none cursor-pointer"
                  >
                    <option value="Apartment">Apartment</option>
                    <option value="Villa">Villa</option>
                    <option value="Independent House">Independent House</option>
                    <option value="Builder Floor">Builder Floor</option>
                  </select>
                </div>
              </div>

              {/* Field 3: BHK */}
              <div className="flex items-center gap-3 px-4 py-2.5 rounded-full bg-white/70 hover:bg-white/90 border border-white/60 transition-colors">
                <BedDouble className="w-5 h-5 text-slate-700 shrink-0" />
                <div className="text-left w-full">
                  <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    BHK
                  </label>
                  <select
                    value={bhk}
                    onChange={(e) => setBhk(e.target.value)}
                    className="w-full bg-transparent text-sm font-semibold text-slate-900 focus:outline-none cursor-pointer"
                  >
                    <option value="1">1 BHK</option>
                    <option value="2">2 BHK</option>
                    <option value="3">3 BHK</option>
                    <option value="4">4 BHK</option>
                    <option value="5">5+ BHK</option>
                  </select>
                </div>
              </div>

              {/* Field 4: Area (sq ft) */}
              <div className="flex items-center gap-3 px-4 py-2.5 rounded-full bg-white/70 hover:bg-white/90 border border-white/60 transition-colors">
                <Maximize2 className="w-5 h-5 text-slate-700 shrink-0" />
                <div className="text-left w-full">
                  <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Area (sq ft)
                  </label>
                  <input
                    type="number"
                    value={area}
                    onChange={(e) => setArea(e.target.value)}
                    placeholder="e.g. 1200"
                    min="200"
                    max="15000"
                    className="w-full bg-transparent text-sm font-semibold text-slate-900 placeholder:text-slate-400 focus:outline-none"
                  />
                </div>
              </div>

              {/* Submit CTA */}
              <button
                type="submit"
                className="w-full h-full min-h-[50px] flex items-center justify-center gap-2 px-6 py-3 rounded-full font-semibold text-white bg-slate-900 hover:bg-slate-800 active:scale-95 shadow-md shadow-slate-900/20 transition-all cursor-pointer"
              >
                <Sparkles className="w-4 h-4 text-amber-400" />
                <span>Estimate Price</span>
                <ArrowRight className="w-4 h-4 text-slate-400" />
              </button>
            </form>
          </div>

        </div>
      </div>
    </section>
  );
}
