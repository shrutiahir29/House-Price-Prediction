/**
 * HouseWise API Configuration
 * Supports local development and cloud production (Vercel + Render)
 */

export const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, "") || "http://127.0.0.1:8000";
