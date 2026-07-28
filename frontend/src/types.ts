/**
 * types.ts — SafeWatch ElderWatch
 * Definisi tipe data dan konstanta statis untuk komponen UI.
 */

// ─── Navigation Tabs ──────────────────────────────────────────────────────────

export type MainTab = 'mobile_app' | 'features' | 'dashboard' | 'setup_guide';

export type SidebarTab =
  | 'home'
  | 'alerts'
  | 'vitals'
  | 'activity'
  | 'notifications'
  | 'settings'
  | 'add_device'
  | 'caregiver';

// ─── Incident / Alert ─────────────────────────────────────────────────────────

export interface Incident {
  id: string;
  title: string;
  location: string;
  time: string;
  severity: 'info' | 'warning' | 'critical';
  details: string;
}

export const MOCK_INCIDENTS: Incident[] = [
  {
    id: 'inc-001',
    title: 'FALL ALERT DETECTED',
    location: 'Living Room',
    time: '2m ago',
    severity: 'critical',
    details: 'High-confidence fall event. Subject is stationary on floor. Immediate response required.',
  },
  {
    id: 'inc-002',
    title: 'Prolonged Inactivity',
    location: 'Bedroom',
    time: '47m ago',
    severity: 'warning',
    details: 'No movement detected for 45 minutes during typical active hours.',
  },
  {
    id: 'inc-003',
    title: 'Night Wandering Detected',
    location: 'Hallway',
    time: '3h ago',
    severity: 'warning',
    details: 'Movement detected at 02:17 AM outside designated safe zones.',
  },
  {
    id: 'inc-004',
    title: 'System Health Check',
    location: 'All Zones',
    time: '6h ago',
    severity: 'info',
    details: 'Routine sensor calibration completed. All cameras operational.',
  },
];

// ─── Room Feeds ───────────────────────────────────────────────────────────────

export interface RoomFeed {
  id: string;
  name: string;
  status: 'live' | 'offline' | 'standby';
}

export const ROOM_FEEDS: RoomFeed[] = [
  { id: 'room-living', name: 'Living Room', status: 'live' },
  { id: 'room-bedroom', name: 'Bedroom', status: 'live' },
  { id: 'room-kitchen', name: 'Kitchen', status: 'standby' },
  { id: 'room-hallway', name: 'Hallway', status: 'live' },
];

// ─── Caregiver Tips ───────────────────────────────────────────────────────────

export interface CaregiverTip {
  id: string;
  category: string;
  title: string;
  content: string;
  icon: string;
}

export const CAREGIVER_TIPS: CaregiverTip[] = [
  {
    id: 'tip-01',
    category: 'Fall Prevention',
    title: 'Secure Rugs & Walkways',
    content:
      'Remove or secure all loose rugs and ensure all walkways are free of clutter. Install non-slip mats in bathrooms and kitchens. Good lighting in all areas, especially at night, is essential.',
    icon: '🏠',
  },
  {
    id: 'tip-02',
    category: 'Medication',
    title: 'Medication Review',
    content:
      'Regularly review all medications with a physician. Some combinations increase fall risk by causing dizziness or drowsiness. Use pill organizers and set reminders for consistent timing.',
    icon: '💊',
  },
  {
    id: 'tip-03',
    category: 'Physical Activity',
    title: 'Balance & Strength Exercises',
    content:
      'Encourage gentle daily exercises focused on balance and leg strength. Activities like Tai Chi, chair yoga, or a short morning walk can significantly reduce fall risk over time.',
    icon: '🧘',
  },
  {
    id: 'tip-04',
    category: 'Vision Care',
    title: 'Regular Eye Exams',
    content:
      'Schedule annual eye exams and ensure prescription glasses are current. Poor vision is a leading contributor to falls. Ensure adequate lighting throughout the home.',
    icon: '👁️',
  },
  {
    id: 'tip-05',
    category: 'Hydration',
    title: 'Prevent Dehydration',
    content:
      'Dehydration can cause dizziness and confusion. Encourage 6–8 glasses of water daily. Keep a water bottle visible and within easy reach. Monitor fluid intake, especially in hot weather.',
    icon: '💧',
  },
  {
    id: 'tip-06',
    category: 'Footwear',
    title: 'Safe & Supportive Footwear',
    content:
      'Ensure the senior wears properly fitting, low-heeled shoes with non-slip soles at all times. Avoid walking in socks alone. Good footwear is one of the simplest fall prevention measures.',
    icon: '👟',
  },
];
