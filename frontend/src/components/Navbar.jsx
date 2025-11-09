import { useState } from "react";
import { Menu, X } from "lucide-react";
import { Link } from "react-router-dom";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="w-full bg-slate-900/90 backdrop-blur-md border-b border-slate-700 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center h-16">
        {/* Logo */}
        <Link
          to="/"
          className="text-2xl font-bold text-blue-400 hover:text-blue-300 transition"
        >
          League Dashboard
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex gap-6 text-gray-300 font-medium">
          <Link
            to="/"
            className="hover:text-blue-400 transition"
          >
            Dashboard
          </Link>
          <Link
            to="/matches"
            className="hover:text-blue-400 transition"
          >
            Matches
          </Link>
          <Link
            to="/settings"
            className="hover:text-blue-400 transition"
          >
            Settings
          </Link>
        </div>

        {/* Mobile Toggle */}
        <button
          className="md:hidden text-gray-300"
          onClick={() => setOpen(!open)}
        >
          {open ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Dropdown */}
      {open && (
        <div className="md:hidden bg-slate-800/95 px-6 py-4 space-y-3">
          <Link
            to="/"
            onClick={() => setOpen(false)}
            className="block text-gray-200 hover:text-blue-400 transition"
          >
            Dashboard
          </Link>
          <Link
            to="/matches"
            onClick={() => setOpen(false)}
            className="block text-gray-200 hover:text-blue-400 transition"
          >
            Matches
          </Link>
          <Link
            to="/settings"
            onClick={() => setOpen(false)}
            className="block text-gray-200 hover:text-blue-400 transition"
          >
            Settings
          </Link>
        </div>
      )}
    </nav>
  );
}
