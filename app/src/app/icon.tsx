import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#17191c",
          borderRadius: 6,
          fontSize: 22,
          fontWeight: 700,
          color: "#F5C518",
          backgroundImage:
            "linear-gradient(135deg, #FFE88C 0%, #F5C518 55%, #B8860B 100%)",
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
        }}
      >
        <span
          style={{
            background: "linear-gradient(135deg, #FFE88C 0%, #F5C518 55%, #B8860B 100%)",
            backgroundClip: "text",
            color: "transparent",
            letterSpacing: "-0.04em",
          }}
        >
          G
        </span>
      </div>
    ),
    { ...size }
  );
}
