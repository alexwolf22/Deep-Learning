function accuracy = ComputeAccuracy(predictions, labels)

    confusion = zeros(36,36);

      % compute statistics only on accumulated pixels
      ok = labels > 1 ;
      numPixels = sum(ok(:)) ;
      confusion = confusion + ...
        accumarray([labels(ok),predictions(ok)],1,[36 36]) ;

      % compute various statistics of the confusion matrix
      pos = sum(confusion,2) ;
      res = sum(confusion,1)' ;
      tp = diag(confusion) ;

      pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
      meanAccuracy = mean(tp ./ max(1, pos)) ;
      meanIntersectionUnion = mean(tp ./ max(1, pos + res - tp)) ;
      
      accuracy = [meanAccuracy pixelAccuracy meanIntersectionUnion];
