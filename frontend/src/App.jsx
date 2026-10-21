import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import HomeShowcaseCards from "./components/HomeShowcaseCards";
import PredictionSection from "./components/PredictionSection";
import ModelComparisonDashboard from "./components/ModelComparisonDashboard";
import AboutSection from "./components/AboutSection";
import Footer from "./components/Footer";
import { INITIAL_METRICS } from "./data/mockData";
import { API_BASE_URL } from "./api/config";

export default function App() {
  const [activeTab, setActiveTab] = useState("home"); // "home" | "predict" | "comparison" | "about"
  const [metrics, setMetrics] = useState(INITIAL_METRICS);
  const [presetData, setPresetData] = useState(null);

  // Fetch live metrics on initial mount
  useEffect(() => {
    async function fetchMetrics() {
      try {
        const res = await fetch(`${API_BASE_URL}/model-metrics`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.models && data.models.length >= 4) {
          const enhanced = data.models.map((m) => {
            const fallback = INITIAL_METRICS.find((f) => f.model_id === m.model_id) || {};
            const baseMae = 47.9403;
            const improvement = m.mae ? ((baseMae - m.mae) / baseMae) * 100 : 0;
            return {
              ...m,
              short_name: fallback.short_name || m.model_name,
              color: fallback.color || "#3b82f6",
              mae_improvement_pct: Math.round(improvement * 100) / 100,
              tag: fallback.tag || "Trained Pipeline",
            };
          });
          setMetrics(enhanced);
        }
      } catch (e) {
        // Fallback initialized
      }
    }
    fetchMetrics();
  }, []);

  // Handle Quick Estimate submission from Hero floating bar
  const handleQuickEstimate = (quickData) => {
    setPresetData({
      City: quickData.city,
      Property_Type: quickData.propertyType,
      BHK: quickData.bhk,
      Bathrooms: Math.max(1, quickData.bhk - (quickData.bhk > 2 ? 0 : 0)),
      Super_Area_SqFt: quickData.area,
      Carpet_Area_SqFt: Math.round(quickData.area * 0.78),
      Floor_Number: 3,
      Total_Floors: 10,
      Age_of_Property: 3,
      Parking: quickData.bhk >= 3 ? 2 : 1,
      Lift_Available: 1,
      Gated_Community: 1,
      Distance_to_Metro_km: 2.0,
      Distance_to_City_Center_km: 8.0,
      Locality_Type: "Premium",
      Furnishing_Status: "Semi-Furnished",
      model_name: "xgboost",
    });
    setActiveTab("predict");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleSelectProperty = (propertyData) => {
    setPresetData(propertyData);
    setActiveTab("predict");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#f8f9fa] text-slate-800">
      
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onStartPredicting={() => {
          setActiveTab("predict");
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      />

      {/* Main Content Areas */}
      <main className="flex-1">
        {activeTab === "home" && (
          <>
            <HeroSection
              onPredictClick={() => {
                setActiveTab("predict");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              onExploreModelsClick={() => {
                setActiveTab("comparison");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              onQuickEstimateSubmit={handleQuickEstimate}
            />

            <HomeShowcaseCards
              metrics={metrics}
              onSelectPropertyForPrediction={handleSelectProperty}
              onOpenComparisonTab={() => {
                setActiveTab("comparison");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
          </>
        )}

        {activeTab === "predict" && (
          <PredictionSection
            key={presetData ? JSON.stringify(presetData) : "default"}
            initialFormData={presetData}
            onOpenComparison={() => {
              setActiveTab("comparison");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}

        {activeTab === "comparison" && (
          <ModelComparisonDashboard
            onStartPredicting={() => {
              setActiveTab("predict");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}

        {activeTab === "about" && (
          <AboutSection />
        )}
      </main>

      {/* Footer */}
      <Footer onNavigate={(tab) => {
        setActiveTab(tab);
        window.scrollTo({ top: 0, behavior: "smooth" });
      }} />

    </div>
  );
}
