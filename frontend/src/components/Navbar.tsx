import { useState } from 'react';
import { Bell, Settings, Menu, X, ShieldAlert } from 'lucide-react';
import { MainTab } from '../types';

interface NavbarProps {
  activeTab: MainTab;
  setActiveTab: (tab: MainTab) => void;
  onTriggerEmergency: () => void;
  isEmergencyActive: boolean;
}

export default function Navbar({
  activeTab,
  setActiveTab,
  onTriggerEmergency,
  isEmergencyActive
}: NavbarProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard' as MainTab, label: 'Dashboard' },
    { id: 'features' as MainTab, label: 'Features' },
    { id: 'mobile_app' as MainTab, label: 'Mobile App' },
    { id: 'setup_guide' as MainTab, label: 'Setup Guide' },
  ];

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-surface/85 backdrop-blur-md border-b border-outline-variant/30 transition-all duration-300">
      <div className="flex justify-between items-center px-6 md:px-16 h-16 max-w-7xl mx-auto">
        {/* Logo and Brand */}
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setActiveTab('mobile_app')}
            className="flex items-center gap-2 cursor-pointer group text-left focus:outline-none"
            id="brand-logo"
          >
            <span className="font-sans text-xl font-extrabold text-primary tracking-tight transition-colors group-hover:text-primary-container">
              Safewatch
            </span>
          </button>
          
          {/* Live indicator badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-0.5 bg-surface-container-high rounded-full border border-outline-variant/35 shadow-sm">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-secondary"></span>
            </span>
            <span className="text-[10px] font-bold uppercase tracking-wider text-on-surface-variant font-mono">
              Live
            </span>
          </div>
        </div>

        {/* Desktop Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-8">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`font-sans text-[15px] font-medium py-1 transition-all relative cursor-pointer focus:outline-none ${
                  isActive
                    ? 'text-primary font-bold border-b-2 border-primary'
                    : 'text-on-surface-variant hover:text-primary'
                }`}
                id={`nav-tab-${item.id}`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Action icons & Emergency Button */}
        <div className="flex items-center gap-2 md:gap-4">
          {/* Icons */}
          <div className="flex items-center gap-1">
            <button 
              className="p-2 text-on-surface-variant hover:bg-surface-container-high/50 hover:text-primary rounded-full transition-all relative focus:outline-none"
              title="Notifications"
              id="notifications-btn"
            >
              <Bell className="w-5 h-5" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-secondary rounded-full"></span>
            </button>
            <button 
              className="p-2 text-on-surface-variant hover:bg-surface-container-high/50 hover:text-primary rounded-full transition-all focus:outline-none"
              title="Settings"
              id="settings-btn"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>

          {/* Emergency call Button */}
          <button
            onClick={onTriggerEmergency}
            className={`px-5 py-2 rounded-full font-sans text-xs font-semibold uppercase tracking-wider shadow-sm active:scale-95 transition-all cursor-pointer focus:outline-none ${
              isEmergencyActive 
                ? 'bg-secondary text-on-secondary animate-pulse'
                : 'bg-primary text-on-primary hover:bg-primary-container'
            }`}
            id="nav-emergency-btn"
          >
            Emergency Call
          </button>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 text-primary hover:bg-surface-container-high/50 rounded-full md:hidden focus:outline-none"
            id="mobile-menu-toggle"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-outline-variant/20 bg-surface px-6 py-4 space-y-3 shadow-lg animate-in fade-in slide-in-from-top-4 duration-200">
          <nav className="flex flex-col space-y-2">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full text-left py-2 px-3 rounded-lg font-sans text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-primary/5 text-primary font-bold'
                      : 'text-on-surface-variant hover:bg-surface-container-low hover:text-primary'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
          
          <div className="pt-2 border-t border-outline-variant/25">
            <button
              onClick={() => {
                onTriggerEmergency();
                setMobileMenuOpen(false);
              }}
              className="w-full flex items-center justify-center gap-2 py-3 bg-secondary text-on-secondary rounded-xl font-sans text-xs font-bold uppercase tracking-wider"
            >
              <ShieldAlert className="w-4 h-4" />
              Trigger Emergency Call
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
