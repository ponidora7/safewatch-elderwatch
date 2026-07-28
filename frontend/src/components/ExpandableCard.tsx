import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Minus, Plus } from 'lucide-react';

interface ExpandableCardProps {
  title: string;
  icon?: React.ReactNode;
  statusDot?: 'green' | 'red' | 'none';
  defaultExpanded?: boolean;
  children: React.ReactNode;
  className?: string;
  headerClassName?: string;
  headerRight?: React.ReactNode;
}

export function ExpandableCard({
  title,
  icon,
  statusDot = 'none',
  defaultExpanded = true,
  children,
  className = '',
  headerClassName = '',
  headerRight,
}: ExpandableCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className={`bg-surface-container-lowest rounded-[2rem] border border-outline-variant/30 overflow-hidden shadow-sm flex flex-col ${className}`}>
      
      {/* Header Area — use div+role to avoid nested <button> error */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => setIsExpanded(!isExpanded)}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setIsExpanded(!isExpanded)}
        className={`w-full px-5 py-4 bg-surface border-b border-outline-variant/20 flex flex-wrap items-center justify-between gap-3 cursor-pointer hover:bg-surface-container-low transition-colors focus:outline-none ${!isExpanded && 'border-b-0'} ${headerClassName}`}
      >
        <div className="flex items-center gap-3">
          {icon && (
            <div className="text-primary opacity-80">
              {icon}
            </div>
          )}
          <span className="font-sans text-sm font-extrabold text-primary uppercase tracking-wider">{title}</span>
          
          {/* Status Indicator */}
          {statusDot !== 'none' && (
            <div className="flex items-center gap-1.5 ml-2">
              <span className={`w-2 h-2 rounded-full ${statusDot === 'green' ? 'bg-emerald-500 pulsing-green-dot' : 'bg-rose-500 pulsing-dot'}`}></span>
              <span className={`text-[9px] font-bold tracking-widest uppercase ${statusDot === 'green' ? 'text-emerald-700' : 'text-rose-600'}`}>
                {statusDot === 'green' ? 'Active' : 'Alert'}
              </span>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-3">
          {headerRight && (
            <div onClick={(e) => e.stopPropagation()} className="cursor-default">
              {headerRight}
            </div>
          )}
          {/* Toggle Icon */}
          <div className="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant transition-colors">
            {isExpanded ? <Minus className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          </div>
        </div>
      </div>

      {/* Expandable Content Area using Framer Motion for smooth height animation */}
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="p-0">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
