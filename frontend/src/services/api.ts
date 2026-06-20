// API Configuration
// Update this URL to point to your FastAPI backend
const API_BASE_URL = (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) || 'http://localhost:8000';


/**
 * Generic API fetch wrapper with error handling and mock data fallback
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  return await response.json();
}

// ============================================================================
// API Functions
// ============================================================================

export async function getMachineData() {
  return apiFetch<MachineData[]>('/api/machine-data');
}

export async function getUtilities() {
  return apiFetch<UtilityData[]>('/api/utilities');
}

export async function getOEEData() {
  return apiFetch<OEEData[]>('/api/oee');
}

export async function getDatabaseRecords(params?: { search?: string; status?: string }) {
  const queryParams = new URLSearchParams();
  if (params?.search) queryParams.append('search', params.search);
  if (params?.status && params.status !== 'all') queryParams.append('status', params.status);

  const query = queryParams.toString();
  const endpoint = `/api/database-records${query ? `?${query}` : ''}`;

  return apiFetch<DataRecord[]>(endpoint);
}

export async function getMachineTimeline(timeRange: 'shift' | 'day' | 'week' | 'month') {
  return apiFetch<TimelineData[]>(`/api/machine-timeline?range=${timeRange}`);
}

export async function getAlerts() {
  return apiFetch<Alert[]>('/api/alerts');
}

export async function getLotAnalytics() {
  return apiFetch<LotAnalytics[]>('/api/analytics/lot');
}

export async function getProductionAnalytics() {
  return apiFetch<ProductionAnalytics[]>('/api/analytics/production');
}

export async function getUtilitiesAnalytics() {
  return apiFetch<UtilitiesAnalytics[]>('/api/analytics/utilities');
}

// ============================================================================
// TypeScript Interfaces
// ============================================================================

export interface MachineData {
  id: string;
  name: string;
  lot1: string;
  lot2: string;
  articleNumber: string;
  totalLength: number; // meters
  status: 'running' | 'idle' | 'maintenance' | 'error';
  lotTime: number;          // minutes
  machineRunningTime: number; // minutes
  speed: number;            // m/min
}

export interface UtilityData {
  id: string;
  name: string;
  type: 'sf' | 'water' | 'air' | 'gas' | 'power';
  processValue: number;
  processUnit: string;
  totalizer: number;
  totalizerUnit: string;
  lotConsumption: number;
  lotConsumptionUnit: string;
  energy?: number;        // Only for EM Power
  energyUnit?: string;
  status: 'normal' | 'warning' | 'critical';
}

export interface OEEData {
  machine_id: string;
  machine_name: string;
  availability: number;
  performance: number;
  quality: number;
  oee: number;
}

export interface DataRecord {
  id: string;
  timestamp: string;
  machineId: string;
  lot1: string;
  lot2: string;
  articleNumber: string;
  totalLength: number;
  speed: number;
  lotTime: number;
  machineRunningTime: number;
  sfConsumption: number;
  waterConsumption: number;
  airConsumption: number;
  gasConsumption: number;
  powerConsumption: number;
  status: string;
}

export interface TimelineData {
  time: string;
  running: number;
  stopped: number;
}

export interface Alert {
  id: string;
  timestamp: string;
  type: string;
  message: string;
  severity: 'critical' | 'warning' | 'info';
  iconName: 'Activity' | 'ThermometerSun' | 'AlertTriangle' | 'Clock';
}

export interface LotAnalytics {
  lot: string;
  speed: number;
  totalLength: number;
}

export interface ProductionAnalytics {
  hour: string;
  rate: number;
  target: number;
}

export interface UtilitiesAnalytics {
  utility: string;
  usage: number;
  cost: number;
}

// Legacy export for compatibility
export type TemperatureAnalytics = LotAnalytics;
export async function getTemperatureAnalytics() {
  return getLotAnalytics();
}