/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import Navbar from './components/Navbar';
import MobileAppPromo from './components/MobileAppPromo';
import FeaturesView from './components/FeaturesView';
import DashboardView from './components/DashboardView';
import SetupGuideView from './components/SetupGuideView';
import EmergencyModal from './components/EmergencyModal';
import { MainTab } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<MainTab>('mobile_app');
  const [emergencyOpen, setEmergencyOpen] = useState(false);

  const handleTriggerEmergency = () => {
    setEmergencyOpen(true);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background font-sans text-on-surface antialiased transition-colors duration-300">
      
      {/* Universal Navbar */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        onTriggerEmergency={handleTriggerEmergency}
        isEmergencyActive={emergencyOpen}
      />

      {/* Main Screen Content Stage */}
      <main className="flex-1">
        {activeTab === 'mobile_app' && (
          <MobileAppPromo 
            onNavigate={setActiveTab} 
            onTriggerEmergency={handleTriggerEmergency} 
          />
        )}
        
        {activeTab === 'features' && (
          <FeaturesView 
            onNavigate={setActiveTab} 
          />
        )}

        {activeTab === 'dashboard' && (
          <DashboardView 
            onTriggerEmergency={handleTriggerEmergency}
            isEmergencyActive={emergencyOpen}
            onNavigate={setActiveTab}
          />
        )}

        {activeTab === 'setup_guide' && (
          <SetupGuideView 
            onNavigate={setActiveTab} 
          />
        )}
      </main>

      {/* Emergency Event Countdown overlay */}
      <EmergencyModal 
        isOpen={emergencyOpen} 
        onClose={() => setEmergencyOpen(false)} 
      />

      {/* Footer (On Landing or non-Dashboard pages) */}
      {activeTab !== 'dashboard' && (
        <footer className="bg-surface-container-low border-t border-outline-variant/30 py-8 px-6 text-center">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4 text-on-surface-variant text-xs font-semibold">
            <p className="font-sans">© 2026 Safewatch Inc. Empathetic Vigilance, clinical precision.</p>
            <div className="flex gap-6">
              <a href="#privacy" className="hover:text-primary transition-colors">Privacy Charter</a>
              <a href="#hipaa" className="hover:text-primary transition-colors">HIPAA Compliance</a>
              <a href="#support" className="hover:text-primary transition-colors">Emergency Support</a>
            </div>
          </div>
        </footer>
      )}

    </div>
  );
}

