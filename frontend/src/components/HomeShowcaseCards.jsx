import React, { useState } from "react";
import { 
  MapPin, BedDouble, Bath, Maximize2, ChevronRight, 
  Cpu, Zap, ShieldCheck, TrendingUp, Sparkles, CheckCircle2 
} from "lucide-react";

export default function HomeShowcaseCards({ metrics, onSelectPropertyForPrediction, onOpenComparisonTab }) {
  const [selectedMetric, setSelectedMetric] = useState("r2");
  const [activePropertyIndex, setActivePropertyIndex] = useState(0);

  const showcaseProperties = [
    {
      id: "pune-villa",
      title: "Modern 3 BHK Villa",
      location: "Pune, Maharashtra",
      image: "/featured_villa.jpg",
      bhk: 3,
      bathrooms: 3,
      area: 1850,
      locality: "Premium Locality",
      features: ["Gated Community", "Private Garden", "Covered Parking"],
      presetData: {
        City: "Pune",
        Property_Type: "Villa",
        Locality_Type: "Premium",
        BHK: 3,
        Bathrooms: 3,
        Super_Area_SqFt: 1850,
        Carpet_Area_SqFt: 1450,
        Floor_Number: 2,
        Total_Floors: 2,
        Age_of_Property: 2,
        Parking: 2,
        Lift_Available: 0,
        Gated_Community: 1,
        Distance_to_Metro_km: 3.5,
        Distance_to_City_Center_km: 12.0,
        Furnishing_Status: "Semi-Furnished",
        model_name: "xgboost",
      }
    },
    {
      id: "mumbai-penthouse",
      title: "Skyline Luxury Penthouse",
      location: "Mumbai, Maharashtra",
      image: "/hero_villa.jpg",
      bhk: 4,
      bathrooms: 4,
      area: 2400,
      locality: "Ultra Premium",
      features: ["Sea View Balcony", "Smart Automation", "3 Parking Slots"],
      presetData: {
        City: "Mumbai",
        Property_Type: "Apartment",
        Locality_Type: "Premium",
        BHK: 4,
        Bathrooms: 4,
        Super_Area_SqFt: 2400,
        Carpet_Area_SqFt: 1950,
        Floor_Number: 28,
        Total_Floors: 35,
        Age_of_Property: 1,
        Parking: 3,
        Lift_Available: 1,
        Gated_Community: 1,
        Distance_to_Metro_km: 0.8,
        Distance_to_City_Center_km: 4.5,
        Furnishing_Status: "Furnished",
        model_name: "xgboost",
      }
    }
  ];

  const currentProp = showcaseProperties[activePropertyIndex];

  // Map metric values for bar chart
  const getMetricData = () => {
    return metrics.map((m) => {
      let val = 0;
      let label = "";
      if (selectedMetric === "r2") {
        val = m.r2;
        label = val.toFixed(2);
      } else if (selectedMetric === "mae") {
        val = m.mae;
        label = `₹${val.toFixed(1)}L`;
      } else if (selectedMetric === "rmse") {
        val = m.rmse;
        label = `₹${val.toFixed(1)}L`;
      }
      return {
        id: m.model_id,
        name: m.short_name || m.display_name,
        fullName: m.display_name,
        value: val,
        label,
        color: m.color,
      };
    });
  };

  const chartData = getMetricData();
  const maxChartVal = Math.max(...chartData.map((d) => d.value), 1.0);

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* CARD 1: Featured Property (4 cols) */}
        <div className="lg:col-span-4 bg-white/90 backdrop-blur-md rounded-3xl p-5 border border-slate-200/80 shadow-lg hover:shadow-xl transition-all flex flex-col justify-between">
          <div>
            {/* Image Thumbnail with Badge */}
            <div className="relative rounded-2xl overflow-hidden aspect-4/3 mb-4 group cursor-pointer"
                 onClick={() => onSelectPropertyForPrediction(currentProp.presetData)}>
              <img
                src={currentProp.image}
                alt={currentProp.title}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-slate-900/80 backdrop-blur-md text-emerald-400 text-xs font-semibold flex items-center gap-1.5 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>FEATURED PROPERTY</span>
              </div>
            </div>

            {/* Location & Title */}
            <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-1">
              <MapPin className="w-3.5 h-3.5 text-slate-400" />
              <span>{currentProp.location}</span>
            </div>
            <h3 className="text-xl font-bold text-slate-900 font-display mb-3 text-left">
              {currentProp.title}
            </h3>

            {/* Specifications Row */}
            <div className="flex items-center justify-between py-2 border-y border-slate-100 text-slate-700 text-xs font-semibold mb-3">
              <div className="flex items-center gap-1.5">
                <BedDouble className="w-4 h-4 text-slate-400" />
                <span>{currentProp.bhk} BHK</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Bath className="w-4 h-4 text-slate-400" />
                <span>{currentProp.bathrooms} Bathrooms</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Maximize2 className="w-4 h-4 text-slate-400" />
                <span>{currentProp.area.toLocaleString()} sq ft</span>
              </div>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <span className="px-2.5 py-1 rounded-md text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-100 flex items-center gap-1">
                🌿 {currentProp.locality}
              </span>
              <span className="px-2.5 py-1 rounded-md text-[11px] font-medium bg-slate-100 text-slate-700">
                Gated Community
              </span>
            </div>
          </div>

          {/* Action Row */}
          <div className="pt-3 flex items-center justify-between border-t border-slate-100">
            <button
              onClick={() => onSelectPropertyForPrediction(currentProp.presetData)}
              className="text-xs font-bold text-slate-900 hover:text-amber-600 flex items-center gap-1 cursor-pointer transition-colors"
            >
              <span>Instant AI Valuation</span>
              <ChevronRight className="w-4 h-4" />
            </button>

            {/* Cycle Property Arrow */}
            <button
              onClick={() => setActivePropertyIndex((prev) => (prev + 1) % showcaseProperties.length)}
              className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 flex items-center justify-center transition-colors cursor-pointer"
              title="Next Property"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* CARD 2: Our Project at a Glance (4 cols) */}
        <div className="lg:col-span-4 bg-white/90 backdrop-blur-md rounded-3xl p-6 border border-slate-200/80 shadow-lg flex flex-col justify-between text-left">
          <div>
            <h3 className="text-xl font-bold text-slate-900 font-display mb-6">
              Our Project at a Glance
            </h3>

            <div className="grid grid-cols-3 gap-3 text-center mb-6">
              {/* Feature 1 */}
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-2xl bg-sky-50 text-sky-600 flex items-center justify-center mb-2.5 shadow-xs">
                  <Cpu className="w-6 h-6" />
                </div>
                <h4 className="text-sm font-bold text-slate-900 leading-tight mb-1">
                  4
                </h4>
                <div className="text-[11px] font-semibold text-slate-700 uppercase tracking-tight">
                  ML Models
                </div>
                <p className="text-[10px] text-slate-500 mt-1 leading-tight">
                  Linear Regression, Decision Tree, Random Forest, XGBoost
                </p>
              </div>

              {/* Feature 2 */}
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-2xl bg-amber-50 text-amber-600 flex items-center justify-center mb-2.5 shadow-xs">
                  <Zap className="w-6 h-6" />
                </div>
                <h4 className="text-sm font-bold text-slate-900 leading-tight mb-1">
                  Real-time
                </h4>
                <div className="text-[11px] font-semibold text-slate-700 uppercase tracking-tight">
                  Estimates
                </div>
                <p className="text-[10px] text-slate-500 mt-1 leading-tight">
                  Get instant price predictions in seconds.
                </p>
              </div>

              {/* Feature 3 */}
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-2.5 shadow-xs">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <h4 className="text-sm font-bold text-slate-900 leading-tight mb-1">
                  Transparent
                </h4>
                <div className="text-[11px] font-semibold text-slate-700 uppercase tracking-tight">
                  Metrics
                </div>
                <p className="text-[10px] text-slate-500 mt-1 leading-tight">
                  MAE, RMSE & R² for fair comparison.
                </p>
              </div>
            </div>
          </div>

          {/* Slogan */}
          <div className="pt-4 border-t border-slate-100 text-center">
            <span className="text-xs font-semibold text-slate-600 italic">
              Data &nbsp;+&nbsp; Machine Learning &nbsp;=&nbsp;{" "}
              <span className="relative font-bold text-slate-900 not-italic">
                Smarter Choices
                <svg className="absolute -bottom-2 left-0 w-full h-2 text-amber-500" viewBox="0 0 100 10" preserveAspectRatio="none">
                  <path d="M0 5 Q 50 12 100 5" fill="transparent" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
              </span>
            </span>
          </div>
        </div>

        {/* CARD 3: Model Performance Comparison (4 cols) */}
        <div className="lg:col-span-4 bg-white/90 backdrop-blur-md rounded-3xl p-6 border border-slate-200/80 shadow-lg flex flex-col justify-between">
          <div>
            {/* Header + Dropdown */}
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-base font-bold text-slate-900 font-display text-left">
                Model Performance Comparison
              </h3>
              
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="text-xs font-semibold text-slate-700 bg-slate-100 border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none cursor-pointer"
              >
                <option value="r2">R² Score</option>
                <option value="mae">MAE (Lakhs)</option>
                <option value="rmse">RMSE (Lakhs)</option>
              </select>
            </div>

            {/* Custom Bar Chart matching reference image */}
            <div className="h-44 w-full flex items-end justify-between gap-3 px-2 pt-6 pb-2 border-b border-slate-200 relative">
              {/* Horizontal Reference Lines */}
              <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-20 pb-2">
                <div className="border-b border-slate-400 w-full" />
                <div className="border-b border-slate-400 w-full" />
                <div className="border-b border-slate-400 w-full" />
                <div className="border-b border-slate-400 w-full" />
              </div>

              {chartData.map((bar) => {
                // Calculate percentage height
                const pct = selectedMetric === "r2" 
                  ? Math.max(15, (bar.value / 1.0) * 100) 
                  : Math.max(20, (bar.value / maxChartVal) * 100);

                return (
                  <div key={bar.id} className="flex-1 flex flex-col items-center h-full justify-end group z-10">
                    {/* Value Badge */}
                    <span className="text-[11px] font-bold text-slate-700 mb-1 group-hover:scale-110 transition-transform">
                      {bar.label}
                    </span>

                    {/* Bar */}
                    <div
                      style={{ 
                        height: `${pct}%`,
                        backgroundColor: bar.color
                      }}
                      className="w-full max-w-[42px] rounded-t-xl transition-all duration-700 ease-out shadow-sm group-hover:brightness-110 cursor-pointer"
                      title={`${bar.fullName}: ${bar.label}`}
                    />
                  </div>
                );
              })}
            </div>

            {/* X-Axis Labels */}
            <div className="grid grid-cols-4 gap-1 pt-2 text-center">
              {chartData.map((bar) => (
                <div key={bar.id} className="text-[10px] font-medium text-slate-600 truncate leading-tight" title={bar.fullName}>
                  {bar.name}
                </div>
              ))}
            </div>
          </div>

          {/* Bottom Link to Full Dashboard */}
          <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
            <span className="text-[11px] text-slate-500">
              {selectedMetric === "r2" ? "Higher score is better" : "Lower error is better"}
            </span>
            <button
              onClick={onOpenComparisonTab}
              className="font-bold text-slate-900 hover:text-amber-600 flex items-center gap-1 cursor-pointer transition-colors"
            >
              <span>Full Analytics</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

      </div>
    </section>
  );
}
