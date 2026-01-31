import { Brain, Shield, Zap, Eye, Microscope, Lock } from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'Advanced AI Models',
    description: 'Ensemble of cutting-edge machine learning models trained on diverse deepfake datasets',
  },
  {
    icon: Shield,
    title: 'Enterprise Security',
    description: 'Bank-level encryption and security protocols protect your sensitive media',
  },
  {
    icon: Zap,
    title: 'Lightning Fast',
    description: 'GPU-accelerated processing analyzes videos in real-time',
  },
  {
    icon: Eye,
    title: 'Detailed Analysis',
    description: 'Frame-by-frame examination with confidence scores and detailed metrics',
  },
  {
    icon: Microscope,
    title: 'Comprehensive Detection',
    description: 'Identifies facial synthesis, lip-sync manipulation, and voice alterations',
  },
  {
    icon: Lock,
    title: 'Privacy First',
    description: 'Your videos are never stored. Complete analysis without data retention',
  },
];

export default function Features() {
  return (
    <section id="features" className="py-20 px-4 border-t border-border">
      <div className="max-w-6xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-balance">
            Industry-Leading Detection
          </h2>
          <p className="text-lg text-foreground/60 max-w-2xl mx-auto">
            Powered by the most advanced deepfake detection technology available
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="p-6 rounded-2xl bg-card/50 border border-border hover:border-accent/50 hover:bg-card/80 transition-smooth group"
            >
              <div className="p-3 w-fit rounded-lg bg-accent/10 group-hover:bg-accent/20 mb-4 transition-smooth">
                <feature.icon className="w-6 h-6 text-accent" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-foreground/60">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
