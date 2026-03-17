/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#08121C",
        mist: "#EFF6F8",
        cyan: "#0EA5A7",
        gold: "#F59E0B",
        coral: "#F97316"
      },
      fontFamily: {
        display: ["Poppins", "ui-sans-serif", "system-ui"],
        body: ["Manrope", "ui-sans-serif", "system-ui"]
      },
      boxShadow: {
        glow: "0 20px 60px rgba(8, 18, 28, 0.14)"
      }
    }
  },
  plugins: []
};
