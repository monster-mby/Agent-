import { create } from 'zustand'

interface UIState {
  sidebarOpen: boolean
  contextPanelOpen: boolean

  toggleSidebar: () => void
  toggleContextPanel: () => void
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  contextPanelOpen: true,

  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  toggleContextPanel: () => set((state) => ({ contextPanelOpen: !state.contextPanelOpen })),
}))
