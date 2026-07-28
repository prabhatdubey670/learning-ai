import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Car Price Intelligence — Live Dashboard",
  description: "Real-time used-car price prediction powered by an XGBoost model (MAE ~$1,081, R² 0.96).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* apply saved theme before paint to avoid flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem('theme');if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
