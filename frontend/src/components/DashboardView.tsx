import { useState, useEffect } from 'react';
import { 
  Home, Camera, HeartPulse, BellRing, Settings, BookOpen, PlusCircle,
  Volume2, VolumeX, Maximize2, Clock, Flame, ChevronRight, ShieldAlert,
  Phone, Check, User, Info, Droplet, Moon, Footprints, ArrowRight, X, AlertTriangle,
  Activity
} from 'lucide-react';
import { SidebarTab, ROOM_FEEDS, MOCK_INCIDENTS, CAREGIVER_TIPS, Incident, CaregiverTip, MainTab } from '../types';
import { WebcamFeed } from './WebcamFeed';
import { ExpandableCard } from './ExpandableCard';
import { supabase } from '../lib/supabase';

interface DashboardViewProps {
  onTriggerEmergency: () => void;
  isEmergencyActive: boolean;
  onNavigate: (tab: MainTab) => void;
}

export default function DashboardView({ 
  onTriggerEmergency, 
  isEmergencyActive,
  onNavigate 
}: DashboardViewProps) {
  // Sidebar states
  const [activeSidebarTab, setActiveSidebarTab] = useState<SidebarTab>('home');
  
  // Living room feed switcher
  const [activeRoomIndex, setActiveRoomIndex] = useState(0);
  const activeRoom = ROOM_FEEDS[activeRoomIndex];
  
  // Real-time ticking clock
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Heart rate jitter simulator
  const [heartRate, setHeartRate] = useState(72);
  useEffect(() => {
    const heartTimer = setInterval(() => {
      setHeartRate(prev => {
        const delta = Math.floor(Math.random() * 5) - 2; // -2 to +2
        const next = prev + delta;
        return next > 90 || next < 60 ? 72 : next;
      });
    }, 3000);
    return () => clearInterval(heartTimer);
  }, []);

  // Camera settings
  const [isMuted, setIsMuted] = useState(true);
  
  // Interactive Caregiver Tip Reader State
  const [selectedTip, setSelectedTip] = useState<CaregiverTip | null>(null);

  // Hydration state
  const [hydrationLiters, setHydrationLiters] = useState(1.2);

  // Settings State
  const [smartZoneRadius, setSmartZoneRadius] = useState(50);
  const [fallSensitivity, setFallSensitivity] = useState(85);
  const [nightHoursEnabled, setNightHoursEnabled] = useState(true);

  // Contact Caregiver dialog state
  const [caregiverModalOpen, setCaregiverModalOpen] = useState(false);

  // Add Device pairing code simulation
  const [addDeviceOpen, setAddDeviceOpen] = useState(false);
  const [pairingCode, setPairingCode] = useState('');
  const [pairedDevices, setPairedDevices] = useState<string[]>(['Home Hub 01', 'Smart Plug Kitchen']);
  const [pairingSuccess, setPairingSuccess] = useState(false);

  // Simulated active incidents list
  const [incidents, setIncidents] = useState<Incident[]>(MOCK_INCIDENTS);
  const [simulatedFallActive, setSimulatedFallActive] = useState(false);

  // Real-time Supabase subscription for incidents logged by backend
  useEffect(() => {
    if (!supabase || !supabase.channel) return;

    const channel = supabase
      .channel('realtime-incidents')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'incidents' },
        (payload: any) => {
          const newRow = payload.new;
          
          const severityMap: Record<string, 'info' | 'warning' | 'critical'> = {
            'info': 'info',
            'warning': 'warning',
            'critical': 'critical',
            'error': 'critical',
            'danger': 'critical',
            'fall': 'critical'
          };
          
          const rawSeverity = (newRow.severity || 'critical').toLowerCase();
          const mappedSeverity = severityMap[rawSeverity] || 'critical';
          
          const newAlert: Incident = {
            id: String(newRow.id || `inc-${Date.now()}`),
            title: newRow.title || 'FALL ALERT DETECTED',
            location: newRow.location || 'Living Room',
            time: 'Just Now',
            severity: mappedSeverity,
            details: newRow.details || 'Sudden high-force posture shift detected by remote AI inference feed.'
          };

          if (mappedSeverity === 'critical') {
            setSimulatedFallActive(true);
          }

          setIncidents(prev => [newAlert, ...prev]);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  // Function to simulate a fall alert
  const handleSimulateFall = () => {
    setSimulatedFallActive(true);
    const newAlert: Incident = {
      id: 'inc-sim',
      title: 'FALL ALERT DETECTED',
      location: 'Living Room',
      time: 'Just Now',
      severity: 'critical',
      details: 'Critical: Sudden rapid posture shift matched fall matrix coordinates in grid sector C.'
    };
    setIncidents([newAlert, ...incidents]);
  };

  const handleDismissSimulatedFall = () => {
    setSimulatedFallActive(false);
    setIncidents(prev => prev.filter(inc => inc.id !== 'inc-sim'));
  };

  // Add a water cup
  const handleAddWater = () => {
    setHydrationLiters(prev => {
      const next = prev + 0.25;
      return next > 3.0 ? 1.2 : parseFloat(next.toFixed(2));
    });
  };

  const handleFrameProcessed = (result: any) => {
    if (result.ai_result?.detected) {
      setSimulatedFallActive(true);
      const newAlert: Incident = {
        id: `inc-${Date.now()}`,
        title: 'FALL DETECTED (WEBCAM)',
        location: 'Living Room',
        time: 'Just Now',
        severity: 'critical',
        details: `Critical: A fall was detected by the live computer vision sensor with ${(result.ai_result.confidence * 100).toFixed(1)}% confidence.`
      };
      setIncidents(prev => {
        if (prev.some(p => p.title.includes('FALL DETECTED') && p.id.startsWith('inc-') && Date.now() - parseInt(p.id.split('-')[1]) < 3000)) {
          return prev;
        }
        return [newAlert, ...prev];
      });
    }
  };

  return (
    <div className="w-full min-h-screen bg-background text-on-surface flex pt-16">
      
      {/* Sidebar Navigation */}
      <aside className="hidden lg:flex flex-col w-64 bg-surface-container-low border-r border-outline-variant/30 shrink-0 p-5 space-y-6">
        
        {/* Sidebar Header: Watch Status */}
        <div className="space-y-3 bg-surface p-4 rounded-2xl border border-outline-variant/35 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-on-primary">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-sans text-xs font-bold text-on-surface-variant uppercase tracking-wider">Residence Alpha</h4>
              <p className="text-xs text-primary font-semibold">Active Monitoring</p>
            </div>
          </div>
          <div className="flex items-center gap-2 pt-2 border-t border-outline-variant/20">
            <span className="w-2.5 h-2.5 bg-green-500 rounded-full pulsing-green-dot"></span>
            <span className="text-[11px] font-sans font-bold text-green-700 uppercase tracking-widest">Vigilance: Active</span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 space-y-1">
          {[
            { id: 'home', label: 'Home Overview', icon: Home },
            { id: 'live_feed', label: 'Live Camera Feed', icon: Camera },
            { id: 'health_data', label: 'Health Metrics', icon: HeartPulse },
            { id: 'incidents', label: 'Incident Logs', icon: BellRing },
            { id: 'settings', label: 'System Settings', icon: Settings },
            { id: 'education', label: 'Education & Tips', icon: BookOpen },
          ].map((item) => {
            const Icon = item.icon;
            const isActive = activeSidebarTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveSidebarTab(item.id as SidebarTab);
                  setSelectedTip(null);
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-sans text-xs font-bold uppercase tracking-wider transition-colors cursor-pointer text-left focus:outline-none ${
                  isActive
                    ? 'bg-primary text-on-primary'
                    : 'text-on-surface-variant hover:bg-surface-container hover:text-primary'
                }`}
                id={`sidebar-tab-${item.id}`}
              >
                <Icon className="w-4.5 h-4.5 shrink-0" />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Add Device Pairing Button */}
        <div className="pt-4 border-t border-outline-variant/25">
          <button
            onClick={() => {
              setAddDeviceOpen(true);
              setPairingSuccess(false);
              setPairingCode('');
            }}
            className="w-full flex items-center justify-center gap-2 py-3 bg-surface hover:bg-surface-container border border-outline-variant/50 rounded-xl font-sans text-xs font-bold text-primary transition-all active:scale-95 focus:outline-none cursor-pointer"
            id="add-device-btn"
          >
            <PlusCircle className="w-4 h-4" />
            Add Safe Device
          </button>
        </div>
      </aside>

      {/* Main Main Content Stage */}
      <main className="flex-1 p-6 md:p-10 max-w-5xl mx-auto space-y-8 overflow-y-auto">
        
        {/* Floating Simulation Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-primary/5 rounded-2xl border border-primary/20 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-xl text-primary">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <p className="font-sans text-xs font-bold text-primary">Demo Sandbox Simulation Tools</p>
              <p className="text-[10px] text-on-surface-variant leading-none">Test Safewatch alerts and trigger countdown panels</p>
            </div>
          </div>
          <div className="flex gap-2">
            {!simulatedFallActive ? (
              <button
                onClick={handleSimulateFall}
                className="px-4 py-2 bg-secondary text-on-secondary hover:bg-secondary-container rounded-xl font-sans text-[10px] font-bold uppercase tracking-wider shadow-sm transition-all cursor-pointer"
              >
                Simulate Fall Alert
              </button>
            ) : (
              <button
                onClick={handleDismissSimulatedFall}
                className="px-4 py-2 bg-green-600 text-white rounded-xl font-sans text-[10px] font-bold uppercase tracking-wider transition-all cursor-pointer"
              >
                Clear Simulated Fall
              </button>
            )}
            <button
              onClick={onTriggerEmergency}
              className="px-4 py-2 border border-secondary text-secondary hover:bg-secondary/5 rounded-xl font-sans text-[10px] font-bold uppercase tracking-wider transition-all cursor-pointer"
            >
              Simulate Emergency Dial
            </button>
          </div>
        </div>

        {/* Fall Warning Notification Toast (Exactly as Screen 1 notification overlay style) */}
        {simulatedFallActive && (
          <div className="p-4 bg-secondary-container/10 border-2 border-secondary rounded-2xl flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 animate-bounce shadow-md">
            <div className="flex items-start gap-3">
              <div className="bg-secondary p-2 rounded-xl text-on-secondary shrink-0">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-sans text-xs font-extrabold text-secondary tracking-wide uppercase">SAFEWATCH NOTIFICATION</span>
                  <span className="w-1.5 h-1.5 bg-secondary rounded-full pulsing-dot"></span>
                </div>
                <h4 className="font-sans text-sm font-bold text-primary">CRITICAL STATUS: Fall Detected in Living Room</h4>
                <p className="text-on-surface-variant text-xs">Unusual high-force posture collapse captured by computer vision sensor grid.</p>
              </div>
            </div>
            <div className="flex gap-2 w-full sm:w-auto">
              <button
                onClick={onTriggerEmergency}
                className="flex-1 sm:flex-none px-4 py-2 bg-secondary text-on-secondary hover:bg-secondary-container rounded-lg font-sans text-[11px] font-bold uppercase tracking-wider transition-all text-center"
              >
                Emergency Dial
              </button>
              <button
                onClick={handleDismissSimulatedFall}
                className="flex-1 sm:flex-none px-4 py-2 bg-surface hover:bg-surface-container-high text-on-surface-variant rounded-lg font-sans text-[11px] font-bold uppercase tracking-wider border border-outline-variant/40 transition-all text-center"
              >
                Dismiss Alert
              </button>
            </div>
          </div>
        )}

        {/* Header Block: Ticking Clock & Status (Exactly like Screen 2) */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-outline-variant/30 pb-6">
          <div className="space-y-1">
            <h1 className="font-sans text-2xl md:text-3xl font-extrabold text-primary tracking-tight">
              Residence Alpha Hub
            </h1>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${simulatedFallActive ? 'bg-secondary pulsing-dot' : 'bg-green-500 pulsing-green-dot'}`}></span>
              <p className={`font-sans text-xs font-bold tracking-wider uppercase ${simulatedFallActive ? 'text-secondary' : 'text-green-700'}`}>
                {simulatedFallActive ? 'CRITICAL ALERT: Hazard Active' : 'System Active: No Hazards Detected'}
              </p>
            </div>
          </div>

          {/* Clock Panel */}
          <div className="flex items-center gap-3 bg-surface p-3 rounded-2xl border border-outline-variant/40 shadow-sm">
            <Clock className="w-5 h-5 text-primary" />
            <div className="text-right">
              <p className="font-mono text-base font-extrabold text-primary leading-none">
                {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </p>
              <p className="text-[9px] uppercase font-bold tracking-widest text-on-surface-variant mt-0.5">
                {time.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}
              </p>
            </div>
          </div>
        </div>

        {/* Render Tab Screens */}
        
        {/* SIDEBAR TAB: HOME OVERVIEW (The primary Bento Dashboard screen) */}
        {activeSidebarTab === 'home' && (
          <div className="space-y-8">
            
            {/* Main Bento Grid Block */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
              
              {/* Live activity Video Card (Span 8) */}
              <ExpandableCard
                title="Live Camera Feed"
                icon={<Camera className="w-5 h-5" />}
                statusDot="green"
                className="md:col-span-8"
                defaultExpanded={true}
                headerRight={
                  <div className="flex bg-surface-container-low p-1 rounded-xl border border-outline-variant/30">
                    {ROOM_FEEDS.map((room, idx) => (
                      <button
                        key={room.id}
                        onClick={() => setActiveRoomIndex(idx)}
                        className={`px-3 py-1.5 rounded-lg font-sans text-[10px] font-bold uppercase tracking-wider transition-colors cursor-pointer ${
                          activeRoomIndex === idx
                            ? 'bg-primary text-on-primary'
                            : 'text-on-surface-variant hover:text-primary'
                        }`}
                      >
                        {room.name}
                      </button>
                    ))}
                  </div>
                }
              >
                {/* Video feed block */}
                <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
                  {activeRoomIndex === 0 ? (
                    <WebcamFeed onFrameProcessed={handleFrameProcessed} />
                  ) : (
                    <img
                      alt={activeRoom.name}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover opacity-85 hover:scale-105 transition-transform duration-[4000ms]"
                      src={activeRoom.image}
                    />
                  )}

                  {/* Red Live pill */}
                  <div className="absolute top-4 left-4 bg-secondary text-on-secondary px-2.5 py-1 rounded-md text-[9px] font-bold uppercase tracking-widest flex items-center gap-1.5 shadow-md">
                    <span className="w-1.5 h-1.5 bg-white rounded-full pulsing-dot"></span>
                    LIVE / REC
                  </div>

                  {/* Room status tag overlay */}
                  <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-md text-white px-3 py-1.5 rounded-xl text-[10px] font-semibold border border-white/10">
                    {activeRoom.name} • {activeRoom.status}
                  </div>

                  {/* Camera audio toggle icons */}
                  <div className="absolute bottom-4 right-4 flex gap-1.5">
                    <button
                      onClick={() => setIsMuted(!isMuted)}
                      className="p-2 bg-black/60 backdrop-blur-md text-white hover:bg-black/80 rounded-xl transition-all border border-white/15 cursor-pointer"
                      title={isMuted ? 'Unmute microphone' : 'Mute microphone'}
                    >
                      {isMuted ? <VolumeX className="w-4 h-4 text-secondary" /> : <Volume2 className="w-4 h-4 text-green-400" />}
                    </button>
                    <button
                      className="p-2 bg-black/60 backdrop-blur-md text-white hover:bg-black/80 rounded-xl transition-all border border-white/15 cursor-pointer"
                      title="Fullscreen view"
                      onClick={() => setActiveSidebarTab('live_feed')}
                    >
                      <Maximize2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </ExpandableCard>

              {/* Quick Access & Recent Alerts (Span 4) */}
              <div className="md:col-span-4 flex flex-col gap-6">
                
                {/* Emergency Card Panel */}
                <ExpandableCard
                  title="Emergency Support"
                  icon={<Phone className="w-5 h-5" />}
                  className="rounded-3xl shadow-sm"
                  defaultExpanded={true}
                >
                  <div className="bg-surface-container-highest p-6 space-y-4">
                    <p className="text-[11px] text-on-surface-variant font-medium leading-relaxed">
                      Immediate connection to first responders or designated emergency contacts.
                    </p>

                  <div className="space-y-2">
                    <button
                      onClick={onTriggerEmergency}
                      className="w-full py-3 bg-secondary text-on-secondary hover:bg-secondary-container rounded-xl font-sans text-xs font-bold uppercase tracking-wider tracking-widest shadow-sm active:scale-95 transition-all cursor-pointer"
                    >
                      EMERGENCY CALL
                    </button>
                    <button
                      onClick={() => setCaregiverModalOpen(true)}
                      className="w-full py-3 bg-surface hover:bg-surface-container border border-outline-variant text-primary rounded-xl font-sans text-xs font-bold uppercase tracking-wider tracking-widest transition-all active:scale-95 cursor-pointer"
                    >
                      Contact Caregiver
                    </button>
                  </div>
                  </div>
                </ExpandableCard>

                {/* Recent Alerts List panel */}
                <ExpandableCard
                  title="Recent Activity"
                  icon={<BellRing className="w-5 h-5" />}
                  className="rounded-3xl shadow-sm flex-1"
                  defaultExpanded={true}
                  headerRight={
                    <button 
                      onClick={() => setActiveSidebarTab('incidents')}
                      className="text-[10px] text-secondary font-bold hover:underline"
                    >
                      All logs
                    </button>
                  }
                >
                  <div className="bg-surface-container-lowest p-5 flex flex-col justify-between h-full">
                    <div className="space-y-3">
                      {incidents.slice(0, 3).map((inc) => {
                        const isCritical = inc.severity === 'critical';
                        const isWarning = inc.severity === 'warning';
                        return (
                          <div key={inc.id} className="flex gap-2.5 items-start">
                            <span className={`p-1.5 rounded-lg shrink-0 ${
                              isCritical ? 'bg-secondary text-on-secondary' : isWarning ? 'bg-amber-100 text-amber-700' : 'bg-surface-container text-primary'
                            }`}>
                              {isCritical ? <ShieldAlert className="w-3.5 h-3.5" /> : isWarning ? <Flame className="w-3.5 h-3.5" /> : <Info className="w-3.5 h-3.5" />}
                            </span>
                            <div className="min-w-0">
                              <p className={`font-sans text-xs font-bold truncate ${isCritical ? 'text-secondary' : 'text-primary'}`}>
                                {inc.title}
                              </p>
                              <p className="text-[9px] text-on-surface-variant truncate">{inc.location} • {inc.time}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </ExpandableCard>

              </div>
            </div>

            {/* Health Snippets Row (4 dynamic cards) */}
            <ExpandableCard
              title="Biometric Health Stream"
              icon={<HeartPulse className="w-5 h-5" />}
              statusDot="green"
              className="w-full"
              defaultExpanded={true}
            >
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-5 bg-surface-container-lowest">
              
              {/* Heart Rate */}
              <div className="bg-surface-container-lowest p-5 rounded-2xl border border-outline-variant/30 shadow-sm space-y-3">
                <div className="flex justify-between items-center text-secondary">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant font-mono">Heart Rate</span>
                  <HeartPulse className="w-5 h-5 animate-pulse text-secondary" />
                </div>
                <div className="space-y-1">
                  <div className="text-2xl font-extrabold text-primary flex items-baseline gap-1">
                    {heartRate} <span className="text-[10px] font-normal text-on-surface-variant">BPM</span>
                  </div>
                  <p className="text-[10px] font-medium text-green-700 bg-green-100/60 px-2 py-0.5 rounded-full inline-block">
                    Normal Range
                  </p>
                </div>
              </div>

              {/* Steps progress */}
              <div className="bg-surface-container-lowest p-5 rounded-2xl border border-outline-variant/30 shadow-sm space-y-3">
                <div className="flex justify-between items-center text-primary">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant font-mono">Steps Walked</span>
                  <Footprints className="w-5 h-5 text-primary" />
                </div>
                <div className="space-y-1.5">
                  <div className="text-2xl font-extrabold text-primary">2,410</div>
                  <div className="w-full bg-surface-container rounded-full h-1.5">
                    <div className="bg-primary h-1.5 rounded-full" style={{ width: '60%' }}></div>
                  </div>
                  <p className="text-[9px] text-on-surface-variant font-medium">Goal: 4,000 steps (60%)</p>
                </div>
              </div>

              {/* Sleep Cycle */}
              <div className="bg-surface-container-lowest p-5 rounded-2xl border border-outline-variant/30 shadow-sm space-y-3">
                <div className="flex justify-between items-center text-primary">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant font-mono">Sleep Quality</span>
                  <Moon className="w-5 h-5 text-primary" />
                </div>
                <div className="space-y-1">
                  <div className="text-2xl font-extrabold text-primary">8h 12m</div>
                  <p className="text-[10px] font-medium text-teal-700 bg-teal-100/60 px-2 py-0.5 rounded-full inline-block">
                    Restful Cycle
                  </p>
                </div>
              </div>

              {/* Hydration: Fully interactive cup adding! */}
              <div className="bg-surface-container-lowest p-5 rounded-2xl border border-outline-variant/30 shadow-sm space-y-3 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center text-primary">
                    <span className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant font-mono">Hydration</span>
                    <Droplet className="w-5 h-5 text-secondary" />
                  </div>
                  <div className="text-2xl font-extrabold text-primary pt-1">
                    {hydrationLiters}L <span className="text-[9px] font-normal text-on-surface-variant">/ 2.0L</span>
                  </div>
                </div>
                <button
                  onClick={handleAddWater}
                  className="w-full py-1.5 bg-primary/5 hover:bg-primary/10 text-primary border border-primary/20 rounded-lg text-[9px] font-bold uppercase tracking-widest transition-all active:scale-95 focus:outline-none cursor-pointer"
                >
                  + Log 1 Cup (0.25L)
                </button>
              </div>

            </div>
            </ExpandableCard>

            {/* Caregiver Reading sliding Tips Section */}
            <ExpandableCard
              title="Caregiver Tips & Resources"
              icon={<BookOpen className="w-5 h-5" />}
              className="w-full"
              defaultExpanded={false}
            >
              <div className="p-5 bg-surface-container-lowest">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {CAREGIVER_TIPS.map((tip) => (
                  <div 
                    key={tip.id} 
                    className="bg-surface-container rounded-2xl p-5 border border-outline-variant/35 flex flex-col justify-between hover:shadow-md transition-all group"
                  >
                    <div className="space-y-3">
                      <div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center text-primary group-hover:scale-105 transition-transform">
                        {tip.iconName === 'home' ? <Home className="w-5 h-5" /> : tip.iconName === 'moon' ? <Moon className="w-5 h-5" /> : <HeartPulse className="w-5 h-5" />}
                      </div>
                      <h4 className="font-sans text-sm font-bold text-primary">{tip.title}</h4>
                      <p className="text-on-surface-variant text-xs leading-relaxed line-clamp-3">
                        {tip.description}
                      </p>
                    </div>

                    <button
                      onClick={() => setSelectedTip(tip)}
                      className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-secondary hover:text-secondary-container transition-all cursor-pointer focus:outline-none"
                    >
                      Read full resource <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
              </div>
            </ExpandableCard>

          </div>
        )}

        {/* SIDEBAR TAB: LIVE FEED (Full camera center with stream switching) */}
        {activeSidebarTab === 'live_feed' && (
          <div className="space-y-6">
            <div className="space-y-2">
              <h2 className="font-sans text-xl font-bold text-primary">Residence Alpha Live Cameras</h2>
              <p className="text-on-surface-variant text-sm">Select any viewport to view details, active motion matrices, and sensor feeds.</p>
            </div>

            <div className="bg-black aspect-video w-full rounded-2xl overflow-hidden relative shadow-md">
              {activeRoomIndex === 0 ? (
                <div className="w-full h-full relative">
                  <WebcamFeed onFrameProcessed={handleFrameProcessed} />
                </div>
              ) : (
                <img
                  alt={activeRoom.name}
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover opacity-80"
                  src={activeRoom.image}
                />
              )}
              <div className="absolute top-6 left-6 bg-secondary text-white px-3 py-1 rounded-md text-[10px] font-bold tracking-widest uppercase flex items-center gap-2 z-20">
                <span className="w-2 h-2 bg-white rounded-full pulsing-dot"></span>
                LIVE FEED SECURE
              </div>
              
              {/* Minimize/Close Button */}
              <button 
                onClick={() => setActiveSidebarTab('home')}
                className="absolute top-6 right-6 bg-black/60 hover:bg-black/80 backdrop-blur-md text-white p-2.5 rounded-full border border-white/20 transition-all z-20 cursor-pointer shadow-lg hover:scale-105"
                title="Minimize Camera"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end bg-black/60 backdrop-blur-md p-4 rounded-xl border border-white/10 text-white z-20 mx-auto max-w-[95%]">
                <div>
                  <h3 className="font-sans text-base font-bold">{activeRoom.name} Overview</h3>
                  <p className="text-xs text-on-surface-variant">Sensor Status: {activeRoom.status}</p>
                </div>
                <div className="text-xs font-mono">
                  {time.toLocaleTimeString()} UTC-5
                </div>
              </div>
            </div>

            {/* Room choice strips */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {ROOM_FEEDS.map((room, idx) => (
                <button
                  key={room.id}
                  onClick={() => setActiveRoomIndex(idx)}
                  className={`p-3.5 rounded-xl border transition-all text-left space-y-2 focus:outline-none cursor-pointer ${
                    activeRoomIndex === idx
                      ? 'bg-primary border-primary text-on-primary'
                      : 'bg-surface-container-lowest border-outline-variant/30 hover:bg-surface-container-low text-on-surface'
                  }`}
                >
                  <p className="font-sans text-xs font-bold">{room.name}</p>
                  <p className={`text-[10px] ${activeRoomIndex === idx ? 'text-on-primary-container' : 'text-on-surface-variant'}`}>
                    {room.status}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* SIDEBAR TAB: HEALTH METRICS */}
        {activeSidebarTab === 'health_data' && (
          <div className="space-y-8">
            <div className="space-y-1">
              <h2 className="font-sans text-xl font-bold text-primary">Biometric Health Stream</h2>
              <p className="text-on-surface-variant text-sm">Longitudinal sleep metrics, ambulatory mobility factors, and cardiovascular rate charts.</p>
            </div>

            {/* Simulated Charts layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Daily Sleep patterns */}
              <div className="bg-surface-container-lowest rounded-3xl p-6 border border-outline-variant/30 shadow-sm space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="font-sans text-sm font-bold text-primary">Restorative Sleep Index</h3>
                  <span className="text-[10px] font-semibold text-teal-700 bg-teal-100 px-2.5 py-0.5 rounded-full uppercase">Optimal</span>
                </div>
                
                {/* Visual bar chart */}
                <div className="h-44 flex items-end gap-3 pt-6 w-full">
                  {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, idx) => {
                    // Heights for days
                    const heights = ['60%', '72%', '65%', '85%', '90%', '78%', '82%'];
                    return (
                      <div key={day} className="flex-1 flex flex-col items-center gap-2">
                        <div className="w-full bg-primary/10 rounded-t-lg relative group h-32 flex items-end">
                          <div 
                            className="w-full bg-primary rounded-t-lg group-hover:bg-primary-container transition-colors"
                            style={{ height: heights[idx] }}
                          ></div>
                        </div>
                        <span className="font-mono text-[9px] text-on-surface-variant font-semibold uppercase">{day}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Heart rate index */}
              <div className="bg-surface-container-lowest rounded-3xl p-6 border border-outline-variant/30 shadow-sm space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="font-sans text-sm font-bold text-primary">Hourly HRV Variability</h3>
                  <span className="text-xs font-mono text-secondary font-bold animate-pulse">{heartRate} BPM</span>
                </div>

                {/* Simulated pulse wave */}
                <div className="h-44 bg-surface-container-low rounded-2xl border border-outline-variant/30 flex items-center justify-center p-4 overflow-hidden relative">
                  <div className="absolute left-0 right-0 h-0.5 bg-outline-variant/20"></div>
                  <svg className="w-full h-24 text-secondary stroke-current fill-none stroke-2" viewBox="0 0 400 100">
                    <path d="M 0 50 L 50 50 L 60 50 L 70 20 L 80 80 L 90 50 L 140 50 L 150 50 L 160 10 L 170 90 L 180 50 L 230 50 L 240 50 L 250 25 L 260 75 L 270 50 L 320 50 L 330 50 L 340 10 L 350 90 L 360 50 L 400 50" />
                  </svg>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* SIDEBAR TAB: INCIDENT LOGS */}
        {activeSidebarTab === 'incidents' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <div className="space-y-1">
                <h2 className="font-sans text-xl font-bold text-primary">Residence Security Audit</h2>
                <p className="text-on-surface-variant text-sm">Comprehensive safety events recorded in the last 72 hours.</p>
              </div>
              <button
                onClick={() => setIncidents(MOCK_INCIDENTS)}
                className="text-xs font-bold text-secondary hover:underline"
              >
                Reset logs
              </button>
            </div>

            <div className="space-y-4">
              {incidents.map((inc) => {
                const isCritical = inc.severity === 'critical';
                const isWarning = inc.severity === 'warning';
                return (
                  <div 
                    key={inc.id}
                    className={`bg-surface-container-lowest p-6 rounded-2xl border transition-all ${
                      isCritical ? 'border-secondary bg-secondary-container/5' : 'border-outline-variant/30'
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <span className={`p-2 rounded-xl shrink-0 ${
                        isCritical ? 'bg-secondary text-on-secondary' : isWarning ? 'bg-amber-100 text-amber-700' : 'bg-surface-container text-primary'
                      }`}>
                        {isCritical ? <ShieldAlert className="w-5 h-5" /> : isWarning ? <Flame className="w-5 h-5" /> : <Info className="w-5 h-5" />}
                      </span>
                      <div className="space-y-1 flex-1">
                        <div className="flex flex-wrap justify-between items-center gap-2">
                          <h4 className={`font-sans text-sm font-extrabold ${isCritical ? 'text-secondary' : 'text-primary'}`}>
                            {inc.title}
                          </h4>
                          <span className={`px-2.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-widest ${
                            isCritical ? 'bg-secondary text-on-secondary' : isWarning ? 'bg-amber-100 text-amber-700' : 'bg-surface-container text-primary'
                          }`}>
                            {inc.severity}
                          </span>
                        </div>
                        <p className="text-[10px] text-on-surface-variant font-semibold">{inc.location} • {inc.time}</p>
                        {inc.details && (
                          <p className="text-xs text-on-surface-variant leading-relaxed pt-2">
                            {inc.details}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* SIDEBAR TAB: SYSTEM SETTINGS (Boundaries configure slide) */}
        {activeSidebarTab === 'settings' && (
          <div className="space-y-8">
            <div className="space-y-1">
              <h2 className="font-sans text-xl font-bold text-primary">Hub Configuration Settings</h2>
              <p className="text-on-surface-variant text-sm">Fine-tune detection thresholds, night tracking ranges, and smart notification zones.</p>
            </div>

            <div className="bg-surface-container-lowest rounded-3xl p-6 md:p-8 border border-outline-variant/30 shadow-sm space-y-6">
              
              {/* Slider 1: Fall computer vision sensitivity */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <h4 className="font-sans text-sm font-bold text-primary">Fall Detection Sensitivity Threshold</h4>
                  <span className="text-xs font-mono font-bold text-secondary">{fallSensitivity}%</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={fallSensitivity}
                  onChange={(e) => setFallSensitivity(Number(e.target.value))}
                  className="w-full h-2 bg-surface-container rounded-lg appearance-none cursor-pointer accent-secondary"
                />
                <p className="text-[10px] text-on-surface-variant leading-relaxed">
                  Higher levels increase detection velocity but may slightly raise false flags. Standard setting is 85%.
                </p>
              </div>

              {/* Slider 2: Smart zone perimeter radius */}
              <div className="space-y-3 pt-4 border-t border-outline-variant/20">
                <div className="flex justify-between items-center">
                  <h4 className="font-sans text-sm font-bold text-primary">Smart Wandering Zone Radius</h4>
                  <span className="text-xs font-mono font-bold text-primary">{smartZoneRadius} meters</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="200"
                  value={smartZoneRadius}
                  onChange={(e) => setSmartZoneRadius(Number(e.target.value))}
                  className="w-full h-2 bg-surface-container rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <p className="text-[10px] text-on-surface-variant leading-relaxed">
                  Triggers immediate warnings if a wearable device or client profile exits this geofence center during quiet hours.
                </p>
              </div>

              {/* Switch button: Night Tracking */}
              <div className="pt-6 border-t border-outline-variant/20 flex justify-between items-center gap-4">
                <div className="space-y-1">
                  <h4 className="font-sans text-sm font-bold text-primary">Night Hours Smart Safeguard</h4>
                  <p className="text-[10px] text-on-surface-variant leading-tight">Apply heightened attention algorithms from 10:00 PM to 6:00 AM.</p>
                </div>
                
                <button
                  onClick={() => setNightHoursEnabled(!nightHoursEnabled)}
                  className={`w-12 h-6 rounded-full relative transition-colors focus:outline-none cursor-pointer ${
                    nightHoursEnabled ? 'bg-primary' : 'bg-surface-container-high'
                  }`}
                >
                  <span className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-all ${
                    nightHoursEnabled ? 'translate-x-6' : 'translate-x-0'
                  }`}></span>
                </button>
              </div>

            </div>
          </div>
        )}

        {/* SIDEBAR TAB: EDUCATION & TIPS READING OVERLAY */}
        {activeSidebarTab === 'education' && (
          <div className="space-y-6">
            <div className="space-y-1">
              <h2 className="font-sans text-xl font-bold text-primary">Elder Care Resource Hub</h2>
              <p className="text-on-surface-variant text-sm">Professional care strategies, home adjustment guides, and medical insights.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {CAREGIVER_TIPS.map((tip) => (
                <div 
                  key={tip.id} 
                  className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/30 flex flex-col justify-between hover:shadow-md transition-all group"
                >
                  <div className="space-y-3">
                    <h3 className="font-sans text-sm font-bold text-primary">{tip.title}</h3>
                    <p className="text-on-surface-variant text-xs leading-relaxed">
                      {tip.description}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedTip(tip)}
                    className="mt-4 text-xs font-bold text-secondary flex items-center gap-1 hover:underline cursor-pointer focus:outline-none"
                  >
                    Open resource paper <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      {/* FULL TIP READING MODAL / SLIDE OVERLAY */}
      {selectedTip && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex justify-end z-50 animate-fade-in">
          <div className="w-full max-w-xl bg-surface min-h-screen p-8 md:p-10 shadow-2xl overflow-y-auto relative flex flex-col justify-between animate-in slide-in-from-right duration-300">
            <div>
              <button 
                onClick={() => setSelectedTip(null)}
                className="absolute top-6 right-6 p-2 bg-surface-container hover:bg-surface-container-high rounded-full text-on-surface-variant cursor-pointer focus:outline-none"
              >
                <X className="w-5 h-5" />
              </button>

              <span className="text-[10px] font-sans font-extrabold text-secondary uppercase tracking-widest">
                {selectedTip.category}
              </span>

              <h2 className="font-sans text-2xl font-extrabold text-primary tracking-tight mt-2 mb-6">
                {selectedTip.title}
              </h2>

              <div className="prose prose-slate text-on-surface-variant text-sm space-y-4 leading-relaxed font-sans">
                {selectedTip.contentMarkdown.split('\n\n').map((paragraph, idx) => {
                  if (paragraph.startsWith('###')) {
                    return <h3 key={idx} className="font-sans text-base font-extrabold text-primary pt-4">{paragraph.replace('###', '').trim()}</h3>;
                  }
                  if (paragraph.startsWith('1.') || paragraph.startsWith('-') || paragraph.startsWith('*')) {
                    return (
                      <div key={idx} className="pl-4 border-l-2 border-secondary/40 space-y-2 py-1">
                        {paragraph.split('\n').map((line, lIdx) => (
                          <p key={lIdx} className="font-medium text-xs text-on-surface">{line.trim()}</p>
                        ))}
                      </div>
                    );
                  }
                  return <p key={idx}>{paragraph}</p>;
                })}
              </div>
            </div>

            <div className="pt-8 border-t border-outline-variant/30 mt-10">
              <button
                onClick={() => setSelectedTip(null)}
                className="w-full py-3 bg-primary text-on-primary rounded-xl font-sans text-xs font-bold uppercase tracking-wider text-center cursor-pointer"
              >
                Done Reading
              </button>
            </div>
          </div>
        </div>
      )}

      {/* PAIRING / ADD DEVICE MODAL */}
      {addDeviceOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-6 animate-fade-in">
          <div className="bg-surface rounded-3xl p-8 max-w-sm w-full border border-outline-variant/30 shadow-2xl relative space-y-6">
            <button
              onClick={() => setAddDeviceOpen(false)}
              className="absolute top-5 right-5 p-2 bg-surface-container hover:bg-surface-container-high rounded-full text-on-surface-variant cursor-pointer focus:outline-none"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="text-center space-y-2">
              <div className="w-12 h-12 bg-primary/10 rounded-2xl flex items-center justify-center text-primary mx-auto">
                <PlusCircle className="w-6 h-6" />
              </div>
              <h3 className="font-sans text-lg font-bold text-primary">Pair Safe Devices</h3>
              <p className="text-xs text-on-surface-variant">Type the code displayed on your Safewatch hardware camera screen.</p>
            </div>

            {pairingSuccess ? (
              <div className="bg-green-50 p-4 rounded-xl border border-green-200 text-center space-y-2">
                <Check className="w-6 h-6 text-green-600 mx-auto" />
                <p className="font-sans text-xs font-bold text-green-800">Connection Successful!</p>
                <p className="text-[10px] text-green-700 leading-none">Your new device is synchronized in the feed switcher.</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant">Activation Pairing Code</label>
                  <input
                    type="text"
                    maxLength={8}
                    placeholder="e.g., SW-8821"
                    value={pairingCode}
                    onChange={(e) => setPairingCode(e.target.value.toUpperCase())}
                    className="w-full px-4 h-12 bg-surface-container-low border border-outline-variant/50 rounded-xl font-mono text-center text-sm font-bold uppercase placeholder:font-sans focus:outline-none focus:border-primary"
                  />
                </div>
                <button
                  onClick={() => {
                    if (pairingCode.trim().length >= 4) {
                      setPairingSuccess(true);
                      setTimeout(() => {
                        setAddDeviceOpen(false);
                      }, 1500);
                    }
                  }}
                  disabled={pairingCode.trim().length < 4}
                  className="w-full py-3 bg-primary text-on-primary disabled:opacity-50 rounded-xl font-sans text-xs font-bold uppercase tracking-wider text-center cursor-pointer active:scale-95 transition-transform"
                >
                  Pair Device
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* CONTACT CAREGIVER DIALOG */}
      {caregiverModalOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-6 animate-fade-in">
          <div className="bg-surface rounded-3xl p-8 max-w-sm w-full border border-outline-variant/30 shadow-2xl relative space-y-6">
            <button
              onClick={() => setCaregiverModalOpen(false)}
              className="absolute top-5 right-5 p-2 bg-surface-container hover:bg-surface-container-high rounded-full text-on-surface-variant cursor-pointer focus:outline-none"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="text-center space-y-2">
              <div className="w-12 h-12 bg-primary/10 rounded-2xl flex items-center justify-center text-primary mx-auto">
                <User className="w-6 h-6" />
              </div>
              <h3 className="font-sans text-lg font-bold text-primary">Residence Alpha Contacts</h3>
              <p className="text-xs text-on-surface-variant">Designated caregiver network contacts for active circles.</p>
            </div>

            <div className="space-y-4">
              <div className="p-3 bg-surface-container-low rounded-xl border border-outline-variant/30 flex justify-between items-center">
                <div>
                  <p className="font-sans text-xs font-bold text-primary">Dr. Emily Stone (MD)</p>
                  <p className="text-[10px] text-on-surface-variant">Primary Medical Advisor</p>
                </div>
                <a 
                  href="tel:5550199"
                  className="p-2 bg-primary text-on-primary rounded-lg hover:bg-primary-container transition-colors"
                >
                  <Phone className="w-4 h-4" />
                </a>
              </div>

              <div className="p-3 bg-surface-container-low rounded-xl border border-outline-variant/30 flex justify-between items-center">
                <div>
                  <p className="font-sans text-xs font-bold text-primary">Sarah Jenkins (Daughter)</p>
                  <p className="text-[10px] text-on-surface-variant">Primary Circle Lead</p>
                </div>
                <a 
                  href="tel:5550188"
                  className="p-2 bg-primary text-on-primary rounded-lg hover:bg-primary-container transition-colors"
                >
                  <Phone className="w-4 h-4" />
                </a>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
